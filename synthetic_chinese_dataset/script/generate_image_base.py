import os
import logging
from six import iteritems, itervalues, string_types
from six.moves import xrange
import threading
from timeit import default_timer
try:
    from queue import Queue, Empty
except ImportError:
    from Queue import Queue, Empty
from gensim import utils

import sys
import random

logger = logging.getLogger(__name__)
         
def Process(samples, inEnlarge=True, num_image=None, callback=None, epochs=1, input_workers = 1,
          batch_images=1, queue_factor=2, report_delay=1.0, min_size=1):

    logging.basicConfig(
        format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s',
        level=logging.INFO)    

    job_tally = 0
    total_images = num_image

    if epochs > 1:
        total_images = num_image and num_image * epochs
        
    def _do_job(sample_batch, workid):
        tally = 0
        raw_tally = len(sample_batch)
        
        for unit in (sample_batch):
            if callback!=None:
                callback(unit, workid)
                tally += 1
        return tally, raw_tally    

    def worker_loop(workid):
        """Train the model, lifting lists of sentences from the job_queue."""
        jobs_processed = 0
        while True:
            job = job_queue.get()
            if job is None:
                progress_queue.put(None)
                break  # no more jobs => quit this worker
            sample_batch, batch_size = job
            tally, raw_tally = _do_job(sample_batch, workid)
            progress_queue.put((tally, raw_tally))  # report back progress
            jobs_processed += 1
        logger.debug("worker exiting, processed %i jobs", jobs_processed)

    def job_producer():
        job_batch, batch_size = [], 0
        pushed_images = 0
        job_no = 0
        
        sample_num = len(samples)
        for _ in xrange(epochs):
            for num in xrange(num_image):
                #sam_index = random.randint(0, sample_num-1) #take care
                sam_index = num % sample_num
                sample = samples[sam_index]
                if sample!=None:
                    if batch_size + 1 <= batch_images:
                        # yes => add it to the current job
                        job_batch.append(sample)
                        batch_size += 1
                    else:
                        # no => submit the existing job
                        logger.debug(
                            "queueing job #%i (%i=?%i images)",
                            job_no, batch_size, len(job_batch))
                        job_no += 1
                        job_queue.put((job_batch, batch_size))

                        # update the progress for the next job
                        pushed_images += len(job_batch)
                        #if pushed_images < total_images:
                        #        progress = 1.0 * pushed_images / total_images
         
                        job_batch, batch_size = [sample], 1

        # add the last job too (may be significantly smaller than batch_images)
        if job_batch:
            logger.debug(
                "queueing job #%i (%i=?%i images)",
                job_no, batch_size, len(job_batch))            
            job_no += 1
            job_queue.put((job_batch, batch_size))
            pushed_images += len(job_batch)

        if job_no == 0:
            logger.warning(
                "train() called with an empty iterator (if not intended, "
                "be sure to provide a corpus that offers restartable "
                "iteration = an iterable)."
            )
            
        total_images = pushed_images
        
        # give the workers heads up that they can finish -- no more work!
        for _ in xrange(input_workers):
            job_queue.put(None)
        logger.info("job loop exiting, total %i jobs", job_no)

    # buffer ahead only a limited number of jobs.. this is the reason we can't simply use ThreadPool :(
    job_queue = Queue(maxsize=queue_factor * input_workers)
    progress_queue = Queue(maxsize=(queue_factor + 1) * input_workers)

    g_workers = [threading.Thread(target=worker_loop, args=(i,)) for i in xrange(input_workers)]
    g_unfinished_worker_count = len(g_workers)
    g_workers.append(threading.Thread(target=job_producer))

    for thread in g_workers:
        thread.daemon = True  # make interrupting the process with ctrl+c easier
        thread.start()

    trained_images_count, raw_image_count = 0, 0
    start, next_report = default_timer() - 0.00001, 1.0

    while g_unfinished_worker_count > 0:
        report = progress_queue.get()  # blocks if workers too slow
        if report is None:  # a thread reporting that it finished
            g_unfinished_worker_count -= 1
            logger.info("worker thread finished; awaiting finish of %i more threads", g_unfinished_worker_count)
            continue
        
        trained_images, raw_images = report
        job_tally += 1

        # update progress stats
        trained_images_count += trained_images  # only words in vocab & sampled
        raw_image_count += raw_images
        
        # log progress once every report_delay seconds
        elapsed = default_timer() - start
        if elapsed >= next_report:
            # words-based progress %
            logger.info(
                "PROGRESS: at %.2f%% images, %.0f images/s, in_qsize %i, out_qsize %i",
                100.0 * raw_image_count / total_images, trained_images_count / elapsed,
                utils.qsize(job_queue), utils.qsize(progress_queue))
            next_report = elapsed + report_delay

    # all done; report the final stats
    
    elapsed = default_timer() - start
    logger.info(
        "training on %i raw images (%i effective images) took %.1fs, %.0f effective images/s",
        raw_image_count, trained_images_count, elapsed, trained_images_count / elapsed)

    # check that the input corpus hasn't changed during iteration
    if total_images and total_images != raw_image_count:
        logger.warning(
            "supplied raw word count (%i) did not equal expected count (%i)", raw_image_count, total_images
        )

    return trained_images_count 
   

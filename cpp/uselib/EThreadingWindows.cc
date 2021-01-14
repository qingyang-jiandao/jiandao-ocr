/**
 * EThreadingWindows.cpp -- Contains all Windows platform specific definitions
 * of interfaces and concrete classes for multithreading support.
 *
 */

#if _MSC_VER

#include "EThreadingWindows.h"

 /** The global synchonization object factory.	*/
FSynchronizeFactoryWin  GSynchronizeFactoryWin;
FSynchronizeFactory*	GSynchronizeFactory = &GSynchronizeFactoryWin;

/**
 * Constructor that zeroes the handle
 */
FEventWin::FEventWin(void)
{
	Event = NULL;
}

/**
 * Cleans up the event handle if valid
 */
FEventWin::~FEventWin(void)
{
	if (Event != NULL)
	{
		CloseHandle(Event);
	}
}

/**
 * Waits for the event to be signaled before returning
 */
void FEventWin::Lock(void)
{
	WaitForSingleObject(Event,INFINITE);
}

/**
 * Triggers the event so any waiting threads are allowed access
 */
void FEventWin::Unlock(void)
{
	PulseEvent(Event);
}

/**
 * Creates the event. Manually reset events stay triggered until reset.
 * Named events share the same underlying event.
 *
 * @param bIsManualReset Whether the event requires manual reseting or not
 * @param InName Whether to use a commonly shared event or not. If so this
 * is the name of the event to share.
 *
 * @return Returns TRUE if the event was created, FALSE otherwise
 */
bool FEventWin::Create(bool bIsManualReset,const TCHAR* InName)
{
	// Create the event and default it to non-signaled

	Event = CreateEvent(NULL,bIsManualReset,0,InName);
	return Event != NULL;
}

/**
 * Triggers the event so any waiting threads are activated
 */
void FEventWin::Trigger(void)
{
	verify(Event);
	SetEvent(Event);
}

/**
 * Resets the event to an untriggered (waitable) state
 */
void FEventWin::Reset(void)
{
	verify(Event);
	ResetEvent(Event);
}

/**
 * Triggers the event and resets the triggered state NOTE: This behaves
 * differently for auto-reset versus manual reset events. All threads
 * are released for manual reset events and only one is for auto reset
 */
void FEventWin::Pulse(void)
{
	verify(Event);
	PulseEvent(Event);
}

/**
 * Waits for the event to be triggered
 *
 * @param WaitTime Time in milliseconds to wait before abandoning the event
 * (DWORD)-1 is treated as wait infinite
 *
 * @return TRUE if the event was signaled, FALSE if the wait timed out
 */
bool FEventWin::Wait(DWORD WaitTime)
{
	verify(Event);
	return WaitForSingleObject(Event,WaitTime) == WAIT_OBJECT_0;
}

/**
* Zeroes its members
*/
FSynchronizeFactoryWin::FSynchronizeFactoryWin(void)
{
}

/**
* Creates a new critical section
*
* @return The new critical section object or NULL otherwise
*/
FCriticalSection* FSynchronizeFactoryWin::CreateCriticalSection(void)
{
	return new FCriticalSection();
}

/**
* Creates a new event
*
* @param bIsManualReset Whether the event requires manual reseting or not
* @param InName Whether to use a commonly shared event or not. If so this
* is the name of the event to share.
*
* @return Returns the new event object if successful, NULL otherwise
*/
FEvent* FSynchronizeFactoryWin::CreateSynchEvent(bool bIsManualReset,
	const TCHAR* InName)
{
	// Allocate the new object
	FEvent* Event = new FEventWin();
	// If the internal create fails, delete the instance and return NULL
	if (Event->Create(bIsManualReset, InName) == FALSE)
	{
		delete Event;
		Event = NULL;
	}
	return Event;
}

/**
* Cleans up the specified synchronization object using the correct heap
*
* @param InSynchObj The synchronization object to destroy
*/
void FSynchronizeFactoryWin::Destroy(FSynchronize* InSynchObj)
{
	delete InSynchObj;
}

#endif


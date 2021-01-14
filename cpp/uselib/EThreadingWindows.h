/**
 * EThreadingWindows.h -- Contains all Windows platform specific definitions
 * of interfaces and concrete classes for multithreading support.
 */

#ifndef _ETHREADING_WINDOWS_H
#define _ETHREADING_WINDOWS_H

#include <Windows.h>
#include "common.h"

#ifndef INFINITE
#define INFINITE ((DWORD)-1)
#endif

// Notify people of the windows dependency.
#if !defined(_WINBASE_) && !defined(_XTL_)
#error EThreadingWindows.h relies on Windows.h/Xtl.h being included ahead of it
#endif

// Make sure version is high enough for API to be defined.
#if !defined(_XTL_) && (_WIN32_WINNT < 0x0403)
#error SetCriticalSectionSpinCount requires _WIN32_WINNT >= 0x0403
#endif

/** Simple base class for polymorphic cleanup */
struct FSynchronize
{
	/** Simple destructor */
	virtual ~FSynchronize(void)
	{
	}
};

/**
 * This class is the abstract representation of a waitable event. It is used
 * to wait for another thread to signal that it is ready for the waiting thread
 * to do some work. Very useful for telling groups of threads to exit.
 */
class FEvent : public FSynchronize
{
public:
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
	virtual bool Create(bool bIsManualReset = FALSE,const TCHAR* InName = NULL) = 0;

	/**
	 * Triggers the event so any waiting threads are activated
	 */
	virtual void Trigger(void) = 0;

	/**
	 * Resets the event to an untriggered (waitable) state
	 */
	virtual void Reset(void) = 0;

	/**
	 * Triggers the event and resets the triggered state (like auto reset)
	 */
	virtual void Pulse(void) = 0;

	/**
	 * Waits for the event to be triggered
	 *
	 * @param WaitTime Time in milliseconds to wait before abandoning the event
	 * (DWORD)-1 is treated as wait infinite
	 *
	 * @return TRUE if the event was signaled, FALSE if the wait timed out
	 */
	virtual bool Wait(DWORD WaitTime = INFINITE) = 0;
};

/**
 * This is the Windows version of a critical section. It uses an aggregate
 * CRITICAL_SECTION to implement its locking.
 */
class FCriticalSection :
	public FSynchronize
{
	/**
	 * The windows specific critical section
	 */
	CRITICAL_SECTION CriticalSection;

public:
	/**
	 * Constructor that initializes the aggregated critical section
	 */
	FORCEINLINE FCriticalSection(void)
	{
		InitializeCriticalSection(&CriticalSection);
		SetCriticalSectionSpinCount(&CriticalSection,4000);
	}

	/**
	 * Destructor cleaning up the critical section
	 */
	FORCEINLINE ~FCriticalSection(void)
	{
		DeleteCriticalSection(&CriticalSection);
	}

	/**
	 * Locks the critical section
	 */
	FORCEINLINE void Lock(void)
	{
		// Spin first before entering critical section, causing ring-0 transition and context switch.
		if( TryEnterCriticalSection(&CriticalSection) == 0 )
		{
			EnterCriticalSection(&CriticalSection);
		}
	}

	/**
	 * Releases the lock on the critical seciton
	 */
	FORCEINLINE void Unlock(void)
	{
		LeaveCriticalSection(&CriticalSection);
	}
};

/**
 * This is the Windows version of an event
 */
class FEventWin : public FEvent
{
	/**
	 * The handle to the event
	 */
	HANDLE Event;

public:
	/**
	 * Constructor that zeroes the handle
	 */
	FEventWin(void);

	/**
	 * Cleans up the event handle if valid
	 */
	virtual ~FEventWin(void);

	/**
	 * Waits for the event to be signaled before returning
	 */
	virtual void Lock(void);

	/**
	 * Triggers the event so any waiting threads are allowed access
	 */
	virtual void Unlock(void);

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
	virtual bool Create(bool bIsManualReset = FALSE,const TCHAR* InName = NULL);

	/**
	 * Triggers the event so any waiting threads are activated
	 */
	virtual void Trigger(void);

	/**
	 * Resets the event to an untriggered (waitable) state
	 */
	virtual void Reset(void);

	/**
	 * Triggers the event and resets the triggered state NOTE: This behaves
	 * differently for auto-reset versus manual reset events. All threads
	 * are released for manual reset events and only one is for auto reset
	 */
	virtual void Pulse(void);

	/**
	 * Waits for the event to be triggered
	 *
	 * @param WaitTime Time in milliseconds to wait before abandoning the event
	 * (DWORD)-1 is treated as wait infinite
	 *
	 * @return TRUE if the event was signaled, FALSE if the wait timed out
	 */
	virtual bool Wait(DWORD WaitTime = (DWORD)-1);
};

/**
* This is the factory interface for creating various synchronization objects.
* It is overloaded on a per platform basis to hide how each platform creates
* the various synchronization objects. NOTE: The malloc used by it must be
* thread safe
*/
class FSynchronizeFactory
{
public:
	/**
	* Creates a new critical section
	*
	* @return The new critical section object or NULL otherwise
	*/
	virtual FCriticalSection* CreateCriticalSection(void) = 0;

	/**
	* Creates a new event
	*
	* @param bIsManualReset Whether the event requires manual reseting or not
	* @param InName Whether to use a commonly shared event or not. If so this
	* is the name of the event to share.
	*
	* @return Returns the new event object if successful, NULL otherwise
	*/
	virtual FEvent* CreateSynchEvent(bool bIsManualReset = FALSE, const TCHAR* InName = NULL) = 0;

	/**
	* Cleans up the specified synchronization object using the correct heap
	*
	* @param InSynchObj The synchronization object to destroy
	*/
	virtual void Destroy(FSynchronize* InSynchObj) = 0;
};

/**
* This is the Windows factory for creating various synchronization objects.
*/
class FSynchronizeFactoryWin : public FSynchronizeFactory
{
public:
	/**
	* Zeroes its members
	*/
	FSynchronizeFactoryWin(void);

	/**
	* Creates a new critical section
	*
	* @return The new critical section object or NULL otherwise
	*/
	virtual FCriticalSection* CreateCriticalSection(void);

	/**
	* Creates a new event
	*
	* @param bIsManualReset Whether the event requires manual reseting or not
	* @param InName Whether to use a commonly shared event or not. If so this
	* is the name of the event to share.
	*
	* @return Returns the new event object if successful, NULL otherwise
	*/
	virtual FEvent* CreateSynchEvent(bool bIsManualReset = FALSE, const TCHAR* InName = NULL);

	/**
	* Cleans up the specified synchronization object using the correct heap
	*
	* @param InSynchObj The synchronization object to destroy
	*/
	virtual void Destroy(FSynchronize* InSynchObj);
};

#endif

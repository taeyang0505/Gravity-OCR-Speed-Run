"""Resident and peak memory of this process, and the memory the system has left, without
third-party packages.

``system()`` is ``(available, total)`` physical memory of the machine; ``relieve()`` gives the
pages of memory-mapped weight files that this process no longer touches back to the system
(Windows working set trim, Linux ``MADV_DONTNEED`` is not needed: the kernel reclaims clean
mapped pages by itself) when the available memory runs low."""

from __future__ import annotations

import ctypes
import os
import sys

MB = 1 << 20


if sys.platform == "win32":
    from ctypes import wintypes

    class _Counters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]  # fmt: skip

    _psapi = ctypes.WinDLL("psapi")
    _kernel = ctypes.WinDLL("kernel32")
    _kernel.GetCurrentProcess.restype = wintypes.HANDLE
    _psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(_Counters), wintypes.DWORD]

    def _counters() -> _Counters:
        c = _Counters()
        c.cb = ctypes.sizeof(c)
        _psapi.GetProcessMemoryInfo(_kernel.GetCurrentProcess(), ctypes.byref(c), c.cb)
        return c

    def rss() -> int:
        """Working set (bytes): what is resident now, mapped file pages included."""
        return int(_counters().WorkingSetSize)

    def peak() -> int:
        return int(_counters().PeakWorkingSetSize)

    def private() -> int:
        """Committed private memory (bytes): allocations, without mapped files."""
        return int(_counters().PrivateUsage)

    def peak_private() -> int:
        return int(_counters().PeakPagefileUsage)

else:
    import resource

    def _status(key: str) -> int:
        try:
            with open(f"/proc/{os.getpid()}/status", encoding="ascii") as f:
                for line in f:
                    if line.startswith(key + ":"):
                        return int(line.split()[1]) * 1024
        except OSError:
            pass
        return 0

    def rss() -> int:
        return _status("VmRSS")

    def peak() -> int:
        v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(v if sys.platform == "darwin" else v * 1024)

    def private() -> int:
        return _status("RssAnon") or rss()

    def peak_private() -> int:
        return _status("VmHWM") or peak()


# ----------------------------------------------------------------------------- system memory
if sys.platform == "win32":

    class _MemStatus(ctypes.Structure):
        _fields_ = [
            ("dwLength", wintypes.DWORD), ("dwMemoryLoad", wintypes.DWORD),
            ("ullTotalPhys", ctypes.c_uint64), ("ullAvailPhys", ctypes.c_uint64),
            ("ullTotalPageFile", ctypes.c_uint64), ("ullAvailPageFile", ctypes.c_uint64),
            ("ullTotalVirtual", ctypes.c_uint64), ("ullAvailVirtual", ctypes.c_uint64),
            ("ullAvailExtendedVirtual", ctypes.c_uint64),
        ]  # fmt: skip

    _kernel.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(_MemStatus)]
    _kernel.GlobalMemoryStatusEx.restype = wintypes.BOOL
    _psapi.EmptyWorkingSet.argtypes = [wintypes.HANDLE]
    _psapi.EmptyWorkingSet.restype = wintypes.BOOL

    def system() -> tuple[int, int]:
        """``(available, total)`` physical memory of the machine in bytes."""
        s = _MemStatus()
        s.dwLength = ctypes.sizeof(s)
        if not _kernel.GlobalMemoryStatusEx(ctypes.byref(s)):
            return 0, 0
        return int(s.ullAvailPhys), int(s.ullTotalPhys)

    def _trim() -> bool:
        return bool(_psapi.EmptyWorkingSet(_kernel.GetCurrentProcess()))

else:

    def system() -> tuple[int, int]:
        """``(available, total)`` physical memory of the machine in bytes."""
        vals = {}
        try:
            with open("/proc/meminfo", encoding="ascii") as f:
                for line in f:
                    k, v = line.split(":", 1)
                    vals[k] = int(v.split()[0]) * 1024
        except (OSError, ValueError):
            pass
        if "MemTotal" in vals:
            return vals.get("MemAvailable", vals.get("MemFree", 0)), vals["MemTotal"]
        try:
            pages, size = os.sysconf("SC_PHYS_PAGES"), os.sysconf("SC_PAGE_SIZE")
            avail = os.sysconf("SC_AVPHYS_PAGES")
            return int(avail * size), int(pages * size)
        except (ValueError, OSError, AttributeError):
            return 0, 0

    def _trim() -> bool:
        return False


RAM_RESERVE_ENV = "GRAVITY_OCR_RAM_RESERVE_MB"
DEFAULT_RAM_RESERVE_MB = 1024


def ram_reserve() -> int:
    """Physical memory to leave to the rest of the system (``GRAVITY_OCR_RAM_RESERVE_MB``)."""
    try:
        return int(os.environ.get(RAM_RESERVE_ENV, DEFAULT_RAM_RESERVE_MB)) * MB
    except ValueError:
        return DEFAULT_RAM_RESERVE_MB * MB


def relieve(force: bool = False) -> bool:
    """When the machine has less than ``ram_reserve()`` or a tenth of its memory available,
    drop this process's working set: the pages of the memory-mapped weights go back to the
    system's standby list (re-read from the file cache on the next touch).  Returns whether it
    trimmed."""
    avail, total = system()
    low = avail < max(ram_reserve(), total // 10) if total else False
    if force or low:
        return _trim()
    return False


# ----------------------------------------------------------------------------- mapped files
if sys.platform == "win32":
    _kernel.VirtualUnlock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    _kernel.VirtualUnlock.restype = wintypes.BOOL


def release_mapped(view, base=None) -> bool:
    """Take the pages of ``view`` – a read-only memory map (``np.memmap``) or an array inside
    one (``base``: the whole map) – out of this process's working set.  They stay in the
    system's file cache (standby: counted as available memory), so the next touch re-reads them
    from RAM, not from the disk.

    Windows: ``VirtualUnlock`` on a range that is not locked removes its pages from the working
    set; Linux / macOS: ``madvise(MADV_DONTNEED)`` on the shared file mapping (page aligned)."""
    if view is None or getattr(view, "nbytes", 0) == 0:
        return False
    if sys.platform == "win32":
        _kernel.VirtualUnlock(ctypes.c_void_p(view.ctypes.data), ctypes.c_size_t(view.nbytes))
        return True  # returns FALSE (ERROR_NOT_LOCKED) by design: the pages are released anyway
    base = view if base is None else base
    raw = getattr(base, "_mmap", None)
    if raw is None or not hasattr(raw, "madvise"):
        return False
    import mmap

    if not hasattr(mmap, "MADV_DONTNEED"):
        return False
    start = view.ctypes.data - base.ctypes.data + int(getattr(base, "offset", 0) or 0)
    page = mmap.PAGESIZE
    lo = start // page * page
    try:
        raw.madvise(mmap.MADV_DONTNEED, lo, min(start + view.nbytes - lo, len(raw) - lo))
    except (ValueError, OSError):
        return False
    return True

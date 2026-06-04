// 测量 N 个空闲 goroutine 的内存占用 (运行时统计 + 进程RSS)
package main

import (
	"fmt"
	"os"
	"runtime"
	"strconv"
	"sync"
)

func rssKB() int {
	data, _ := os.ReadFile("/proc/self/status")
	for _, line := range splitLines(string(data)) {
		if len(line) > 6 && line[:6] == "VmRSS:" {
			var v int
			fmt.Sscanf(line[6:], "%d", &v)
			return v
		}
	}
	return -1
}
func splitLines(s string) []string {
	var out []string
	start := 0
	for i := 0; i < len(s); i++ {
		if s[i] == '\n' {
			out = append(out, s[start:i])
			start = i + 1
		}
	}
	return out
}

func main() {
	n, _ := strconv.Atoi(os.Args[1])
	runtime.GC()
	var m0 runtime.MemStats
	runtime.ReadMemStats(&m0)
	base := rssKB()

	var wg sync.WaitGroup
	start := make(chan struct{})
	wg.Add(n)
	for i := 0; i < n; i++ {
		go func() {
			<-start // 全部 goroutine 在此挂起
			wg.Done()
		}()
	}
	// 等待调度器把所有 goroutine 创建并挂起
	for runtime.NumGoroutine() < n+1 {
	}
	runtime.GC()
	var m1 runtime.MemStats
	runtime.ReadMemStats(&m1)
	peak := rssKB()

	heapPerUnit := float64(m1.HeapAlloc-m0.HeapAlloc) / float64(n) / 1024.0
	rssPerUnit := float64(peak-base) / float64(n)
	close(start)
	wg.Wait()
	fmt.Printf("{\"mode\":\"goroutine\",\"n\":%d,\"kb_per_unit_heap\":%.4f,\"kb_per_unit_rss\":%.4f}\n",
		n, heapPerUnit, rssPerUnit)
}

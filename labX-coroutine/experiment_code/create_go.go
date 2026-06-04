package main
import("fmt";"sync";"time")
func main(){
  const n=1000000
  var wg sync.WaitGroup
  wg.Add(n)
  t0:=time.Now()
  for i:=0;i<n;i++{ go func(){wg.Done()}() }
  wg.Wait()
  us:=float64(time.Since(t0).Microseconds())/float64(n)
  fmt.Printf("{\"goroutine_us_per_create\":%.4f,\"n\":%d}\n",us,n)
}

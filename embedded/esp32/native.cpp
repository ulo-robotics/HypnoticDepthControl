// the same header on the host CPU (x86-64, glibc libm), for ctypes
#include <memory>
#include "lethe_infer.hpp"
struct Handle { lethe::Model m; std::unique_ptr<lethe::Policy<float>> f; std::unique_ptr<lethe::Policy<double>> d; };
extern "C" void* lethe_new(const char* path, int dbl) {
  auto* h = new Handle{lethe::Model::load(path), nullptr, nullptr};
  if (dbl) h->d.reset(new lethe::Policy<double>(h->m)); else h->f.reset(new lethe::Policy<float>(h->m));
  return h;
}
extern "C" float lethe_step(void* p, int k, float y, float t, float tp) {
  auto* h = (Handle*)p;
  if (h->f) { if (k == 0) h->f->reset(); return h->f->step(y, t, tp); }
  if (k == 0) h->d->reset();
  return (float)h->d->step(y, t, tp);
}

/* This file was automatically generated by CasADi.
   The CasADi copyright holders make no ownership claim of its contents. */
#ifdef __cplusplus
extern "C" {
#endif

/* How to prefix internal symbols */
#ifdef CASADI_CODEGEN_PREFIX
  #define CASADI_NAMESPACE_CONCAT(NS, ID) _CASADI_NAMESPACE_CONCAT(NS, ID)
  #define _CASADI_NAMESPACE_CONCAT(NS, ID) NS ## ID
  #define CASADI_PREFIX(ID) CASADI_NAMESPACE_CONCAT(CODEGEN_PREFIX, ID)
#else
  #define CASADI_PREFIX(ID) pendulum_ode_expl_ode_hess_ ## ID
#endif

#include <math.h>

#ifndef casadi_real
#define casadi_real double
#endif

#ifndef casadi_int
#define casadi_int int
#endif

/* Add prefix to internal symbols */
#define casadi_f0 CASADI_PREFIX(f0)
#define casadi_s0 CASADI_PREFIX(s0)
#define casadi_s1 CASADI_PREFIX(s1)
#define casadi_s2 CASADI_PREFIX(s2)
#define casadi_s3 CASADI_PREFIX(s3)
#define casadi_s4 CASADI_PREFIX(s4)
#define casadi_s5 CASADI_PREFIX(s5)

/* Symbol visibility in DLLs */
#ifndef CASADI_SYMBOL_EXPORT
  #if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
    #if defined(STATIC_LINKED)
      #define CASADI_SYMBOL_EXPORT
    #else
      #define CASADI_SYMBOL_EXPORT __declspec(dllexport)
    #endif
  #elif defined(__GNUC__) && defined(GCC_HASCLASSVISIBILITY)
    #define CASADI_SYMBOL_EXPORT __attribute__ ((visibility ("default")))
  #else
    #define CASADI_SYMBOL_EXPORT
  #endif
#endif

static const casadi_int casadi_s0[6] = {2, 1, 0, 2, 0, 1};
static const casadi_int casadi_s1[9] = {2, 2, 0, 2, 4, 0, 1, 0, 1};
static const casadi_int casadi_s2[5] = {1, 1, 0, 1, 0};
static const casadi_int casadi_s3[3] = {0, 0, 0};
static const casadi_int casadi_s4[7] = {3, 1, 0, 3, 0, 1, 2};
static const casadi_int casadi_s5[10] = {6, 1, 0, 6, 0, 1, 2, 3, 4, 5};

/* pendulum_ode_expl_ode_hess:(i0[2],i1[2x2],i2[2],i3[2],i4,i5[])->(o0[3],o1[6]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a2, a3, a4;
  a0=arg[0]? arg[0][0] : 0;
  a1=cos(a0);
  a2=1.4715000000000000e+00;
  a3=2.2222222222222221e+01;
  a4=arg[3]? arg[3][1] : 0;
  a3=(a3*a4);
  a2=(a2*a3);
  a1=(a1*a2);
  if (res[0]!=0) res[0][0]=a1;
  a1=arg[3]? arg[3][0] : 0;
  if (res[0]!=0) res[0][1]=a1;
  if (res[0]!=0) res[0][2]=a3;
  a3=arg[1]? arg[1][0] : 0;
  a0=sin(a0);
  a1=(a0*a3);
  a1=(a2*a1);
  a3=(a3*a1);
  a3=(-a3);
  if (res[1]!=0) res[1][0]=a3;
  a3=arg[1]? arg[1][2] : 0;
  a4=(a3*a1);
  a4=(-a4);
  if (res[1]!=0) res[1][1]=a4;
  a4=arg[2]? arg[2][0] : 0;
  a1=(a4*a1);
  a1=(-a1);
  if (res[1]!=0) res[1][2]=a1;
  a1=(a0*a3);
  a1=(a2*a1);
  a3=(a3*a1);
  a3=(-a3);
  if (res[1]!=0) res[1][3]=a3;
  a1=(a4*a1);
  a1=(-a1);
  if (res[1]!=0) res[1][4]=a1;
  a0=(a0*a4);
  a2=(a2*a0);
  a4=(a4*a2);
  a4=(-a4);
  if (res[1]!=0) res[1][5]=a4;
  return 0;
}

CASADI_SYMBOL_EXPORT int pendulum_ode_expl_ode_hess(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int pendulum_ode_expl_ode_hess_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int pendulum_ode_expl_ode_hess_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void pendulum_ode_expl_ode_hess_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int pendulum_ode_expl_ode_hess_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void pendulum_ode_expl_ode_hess_release(int mem) {
}

CASADI_SYMBOL_EXPORT void pendulum_ode_expl_ode_hess_incref(void) {
}

CASADI_SYMBOL_EXPORT void pendulum_ode_expl_ode_hess_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int pendulum_ode_expl_ode_hess_n_in(void) { return 6;}

CASADI_SYMBOL_EXPORT casadi_int pendulum_ode_expl_ode_hess_n_out(void) { return 2;}

CASADI_SYMBOL_EXPORT casadi_real pendulum_ode_expl_ode_hess_default_in(casadi_int i){
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* pendulum_ode_expl_ode_hess_name_in(casadi_int i){
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    case 4: return "i4";
    case 5: return "i5";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* pendulum_ode_expl_ode_hess_name_out(casadi_int i){
  switch (i) {
    case 0: return "o0";
    case 1: return "o1";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* pendulum_ode_expl_ode_hess_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    case 2: return casadi_s0;
    case 3: return casadi_s0;
    case 4: return casadi_s2;
    case 5: return casadi_s3;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* pendulum_ode_expl_ode_hess_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s4;
    case 1: return casadi_s5;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int pendulum_ode_expl_ode_hess_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 6;
  if (sz_res) *sz_res = 2;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif

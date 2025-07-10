#import "template/main.typ": *

#show: template.with(
  title: [Eigendecomposition and Power~Iteration],
  author: "Giap Bui-Huy",
  date: datetime(
    year: 2025,
    month: 7,
    day: 9,
  ),
  toc: true,
)

= Introduction

This article briefly explains matrix eigen(value) decomposition, which will be
abbreviated as EVD. The EVD of a matrix is a factorization of an $n times n$
matrix $bold(A)$ into $bold(P) bold(D) bold(P)^(-1)$, where $bold(P) =
(bold(v)_1, bold(v)_2, ..., bold(v)_n)$ are called the eigenvectors of
$bold(A)$, and $bold(D)$ is a diagonal matrix:

$
bold(D) = mat(
  lambda_1, 0, ..., 0;
  0, lambda_2, ..., 0;
  dots.v, dots.v, dots.down, dots.v;
  0, 0, ..., lambda_n;
)
$

The diagonal entries $lambda_1, lambda_2, ..., lambda_n$ of $bold(D)$ are
called the eigenvalues of $bold(D)$. An eigenvector $bold(v)_i$ and its
corresponding eigenvalue $lambda_i$ is connected through the following formula:

$
bold(A) bold(v)_i = lambda_i bold(v)_i
$

Beause of this relationship, the eigenvectors and eigenvalues of a matrix
provides many insights into the matrix, such as the general scale and direction
of linear transformations and the spread of multivariate datasets. Because of
this, EVD and its derivatives --- spectral decomposition and singular value
decomposition --- are valuable tools in the age of data science and machine
learning.

This article is an informal exploration of the connection between EVD and
matrix power. While using EVD for matrix power is a popular application, what's
interesting is going the other way around. This gives us the Power Iteration
method, a simple and effective way to find the domiance eigenvector. This
method can be extended to find the equivalent eigenvalue, increasing the
precision of a rough eigenvector/eigenvalue approximation, and is the basis of
more advanced EVD algorithms such as the QR algorithm.

= Use the EVD to compute matrix power

Raising a matrix to an integer power, or computing $bold(A)^k$ with $k in ZZ$
is difficult for human, but EVD is _relatively_ easy. So it's natural to turn
to EVD to compute matrix powers. The reason for this is that $bold(D)$ is
diagonal, and it can be trivially proven with induction that:

$
bold(D)^k = mat(
  lambda_1^k, 0, ..., 0;
  0, lambda_2^k, ..., 0;
  dots.v, dots.v, dots.down, dots.v;
  0, 0, ..., lambda_n^k;
)
$

This also allows us to extend the concept of matrix power to real numbers, but
that's outside the scope of this article. For $k in ZZ$, we can write $A^k$ as
repeated multiplications:

$
bold(A)^k = underbrace(bold(A) bold(A) ... bold(A), k "times")
$

Then substitue $bold(A)$ with its EVD $bold(P) bold(D) bold(P)^(-1)$, and
cancel out every $bold(P)$ and $bold(P)^(-1)$ pairs.

$
bold(A)^k
  &= bold(P) bold(D) underbrace(bold(P)^(-1) bold(P), bold(I)_n) bold(D) ... bold(D) bold(P)^(-1) \
  &= bold(P) bold(D)^k bold(P)^(-1) \
  &= mat(
  bar.v, bar.v, , bar.v;
  bold(v)_1, bold(v)_2, ..., bold(v)_n;
  bar.v, bar.v, , bar.v;
) mat(
  lambda_1^k, 0, ..., 0;
  0, lambda_2^k, ..., 0;
  dots.v, dots.v, dots.down, dots.v;
  0, 0, ..., lambda_n^k;
) mat(
  bar.v, bar.v, , bar.v;
  bold(v)_1, bold(v)_2, ..., bold(v)_n;
  bar.v, bar.v, , bar.v;
)^(-1)
$

Then we can multiply these matrices together to get the closed form expression
for $A^k$. Or at least, we can take the EVD of a matrix $bold(A)$ to inspect
the growth rate of $bold(A)^k$.

== Example: Fibonacci sequence

To better illustrate this concept, let's consider the problem of finding the
$n$-th Fibonacci number. The Fibonacci sequence is defined by a recurrence
relation as follow:

$
F_0 &= 0 \
F_1 &= 1 \
F_n &= F_(n - 1) + F_(n - 2) \
$

We can rewrite the case of $n >= 2$ as a linear system of equation to match the
number of unknowns and the number of linear coefficients:

$
cases(
  F_(n + 1) &= bold(1) F_n + bold(1) F_(n - 1),
  F_(n)     &= bold(1) F_n + bold(0) F_(n - 1),
)
$

Which we can factor out as a matrix-vector multiplication, and remove the
recurrence by converting it into a matrix power expression:

$
vec(F_(n + 1), F_n)
  &= mat(1, 1; 1, 0) vec(F_n, F_(n - 1))
  = mat(1, 1; 1, 0)^2 vec(F_(n - 1), F_(n - 2)) \
  &= mat(1, 1; 1, 0)^k vec(F_(n - k + 1), F_(n - k))
  = mat(1, 1; 1, 0)^n vec(F_1, F_0) \
  &= mat(1, 1; 1, 0)^n vec(1, 0) \
$ <fib-power>

Technically we already have a closed form solution, but that's for $(F_(n + 1),
F_n)^T$, and contains matrices which people don't like for some reasons, so
we can solve for the $F_n$ entry of the matrix using EVD method as described
above:

$
& mat(1, 1; 1, 0) = 
  mat(phi, 1-phi; 1, 1)
  mat(phi, 0; 0, 1 - phi)
  mat(phi, 1-phi; 1, 1)^(-1) \
& "where" phi = (1 + sqrt(5))/2 "is the golden ratio" \
& mat(1, 1; 1, 0)^n =
  mat(phi, 1-phi; 1, 1)
  mat(phi^n, 0; 0, (1 - phi)^n)
  mat(phi, 1-phi; 1, 1)^(-1) \
& vec(F_(n + 1), F_n) =
  mat(phi, 1-phi; 1, 1)
  mat(phi^n, 0; 0, (1 - phi)^n)
  mat(phi, 1-phi; 1, 1)^(-1) vec(1, 0) \
$

If you carry out the multiplication, extract the $F_n$ component and simplify,
you'll get:

$
F_n = (phi^n - (1 - phi)^n)/(2 phi - 1)
$

Which is essentially the same the classical Binet's formula for the Fibonacci
sequence.

== Practical considerations <practical-consideration>

Of course, if you want to actually know the numerical value of matrix power,
taking the EVD and raising the power of the eigenvalues is far from the best
way of doing it. EVD is numerically expensive and does not guarantee to exist
for non-symmetric matrices. This method is only useful for finding closed-form
solution, which I'd argue that it's only useful for proofs, symbolic
manipulations, and extension to real exponents. The Fibonacci example also
shown how to convert linear recurrences into matrix power which is useful for
later examples.

In practice, for computing $A^k$, we can just use repeated multiplication, or
even better, exponentation by squaring. If you use numpy, there's the
`np.linalg.matrix_power` function which is an efficient implementation for the
latter method. It can also make use of the fact that the matrix entries are
integers and avoid the overhead and rounding error of floating point
arithmetic. This prompts a question: If the EVD can be used to compute matrix
power, can we do the opposite and use matrix power to compute the EVD of a
matrix?

= Power iteration

The Power Iteration is a simple algorithm for finding the eigenvector with the
largest corresponding eigenvalue of a matrix. The algorithm can be extended for
more eigenvectors, but even the largest eigenvector alone is already useful. We
can use it to find the direction of maximum variance of a dataset.

The method of Power Iteration is usually described and implemented as follow:

- Start with an initial random vector $bold(b)_0$
- At step $k$, update the vector to get $bold(b)_(k + 1)$ using the following
  recurrence relation:
  $
  bold(b)_(k + 1) = (bold(A) bold(b)_k)/norm(bold(A) bold(b)_k)_2
  $ <power-recurrence>

Here's an example implementation in Python with NumPy:

```python
import numpy as np

type NPMat = np.ndarray[tuple[int, int], np.dtype[np.float64]]
type NPVec = np.ndarray[tuple[int], np.dtype[np.float64]]
type NPRng = np.random.Generator

def power_iteration(A: NPMat, n_iter: int, rng: NPRng) -> NPVec:
  n, m = A.shape
  assert m == n, f"A ({m}×{n}) is non-square"

  b: NPVec = rng.standard_normal((n,))

  for _ in range(n_iter):
    Ab = A @ b
    b = Ab / np.linalg.norm(Ab) 

  return b
```

The algorithm is relatively straightforward to implement and execute with no
edge-cases to handle. It also work with black-box matrices, where you pass in
$f: RR^n -> RR^n$ instead of the explicit $RR^(n times n)$ matrix which enhance
extensibility.

== Proof of convergence <power-proof>

We will prove that the algorithm will converge to a largest eigenvector, if we
order the eigenvalue so that their absolute value is in ascending order
$abs(lambda_1) >= abs(lambda_2) >= ... >= abs(lambda_n)$, we'll have to show
that if $display(lim_(k -> oo) bold(b)_k = bold(v))$, then $bold(A) bold(v) =
lambda_1 bold(v)$. For that, let's take a better look at the recurrence.
Similar with the Fibonacci example, we can try to factor it into a matrix power
expression like we did in @fib-power.

$
bold(b)_(k + 1)
  &= (bold(A) bold(b)_k)/norm(bold(A) bold(b)_k)_2
  = (bold(A) (bold(A) bold(b)_(k - 1))/norm(bold(A) bold(b)_(k - 1))_2)/
    norm(bold(A) (bold(A) bold(b)_(k - 1))/norm(bold(A) bold(b)_(k - 1))_2)_2
  = cancel((1/norm(bold(A) bold(b)_(k - 1))_2)/(1/norm(bold(A) bold(b)_(k - 1))_2))
    (bold(A)^2 bold(b)_(k - 1))/norm(bold(A)^2 bold(b)_(k - 1))_2
  = (bold(A)^(k + 1) bold(b)_0)/norm(bold(A)^(k + 1) bold(b)_0)_2
$

If we substitute $k + 1$ with $k$, we get the following formula for
$bold(b)_k$, which is the matrix power of $bold(A)$ applied to the initial
vector $bold(b)_0$.

$
bold(b)_k = (bold(A)^k bold(b)_0)/norm(bold(A)^k bold(b)_0)_2
$ <power-factor>

This is much easier to analyze the asymptotic as $k$ approaches $oo$, we
can compute the EVD of $A$ (assuming that it exists), and use the following
substitutions:

$
bold(A)^k &= bold(P) bold(D)^k bold(P)^(-1) = mat(
  bar.v, bar.v, , bar.v;
  bold(v)_1, bold(v)_2, ..., bold(v)_n;
  bar.v, bar.v, , bar.v;
) mat(
  lambda_1^k, 0, ..., 0;
  0, lambda_2^k, ..., 0;
  dots.v, dots.v, dots.down, dots.v;
  0, 0, ..., lambda_n^k;
) mat(
  bar.v, bar.v, , bar.v;
  bold(v)_1, bold(v)_2, ..., bold(v)_n;
  bar.v, bar.v, , bar.v;
)^(-1) \

accent(bold(b), tilde)
  &= mat(
    accent(bold(b), tilde)_1,
    accent(bold(b), tilde)_2,
    ...,
    accent(bold(b),
    tilde)_n
  )^T = bold(P)^(-1) bold(b)_0 => bold(b)_0 = bold(P) accent(bold(b)_0, tilde) \
$

Then, expanding $bold(A)^k bold(b)_0$ gives us:

$
bold(A)^k bold(b)_0
  = bold(P) bold(D)^k bold(P)^(-1) bold(P) accent(bold(b), tilde)
  = bold(P) bold(D)^k accent(bold(b), tilde) = mat(
  bar.v, bar.v, , bar.v;
  bold(v)_1, bold(v)_2, ..., bold(v)_n;
  bar.v, bar.v, , bar.v;
) mat(
  lambda_1^k, 0, ..., 0;
  0, lambda_2^k, ..., 0;
  dots.v, dots.v, dots.down, dots.v;
  0, 0, ..., lambda_n^k;
) vec(
  accent(bold(b), tilde)_1,
  accent(bold(b), tilde)_2,
  dots.v, accent(bold(b),
  tilde)_n
)
$

Carrying out the matrix multiplications, we get a linear combination of
eigenvectors:

$
bold(A)^k bold(b)_0
  &= mat(
  bar.v, bar.v, , bar.v;
  bold(v)_1, bold(v)_2, ..., bold(v)_n;
  bar.v, bar.v, , bar.v;
) vec(
  lambda_1^k accent(bold(b), tilde)_1,
  lambda_2^k accent(bold(b), tilde)_2,
  dots.v,
  lambda_n^k accent(bold(b), tilde)_n
) \
  &= lambda_1^k accent(bold(b), tilde)_0 bold(v)_1
  + lambda_2^k accent(bold(b), tilde)_2 bold(v)_2
  + ...
  + lambda_n^k accent(bold(b), tilde)_n bold(v)_n \
  &= lambda_1^k (accent(bold(b), tilde)_0 bold(v)_1
  + (lambda_2/lambda_1)^k accent(bold(b), tilde)_2 bold(v)_2
  + ...
  + (lambda_n/lambda_1)^k accent(bold(b), tilde)_n bold(v)_n) \
$

Let $m$ be the number of repeated top eigenvalues, we have:

$
abs(lambda_1) = ... = abs(lambda_m) > abs(lambda_(m + 1)) >= ... >= abs(lambda_n)
$

Assuming $exists i (1 <= i <= m)$ such that $accent(bold(b), tilde)_i != 0$, we
have the following limit:

$
lim_(k -> oo) (lambda_i/lambda_1)^k = cases(
  1 "if" i <= m,
  0 "if" i > m,
) \

lim_(k -> oo) bold(A)^k bold(b)_0
  = sum_(i = 1)^m accent(bold(b), tilde)_i lambda_1^k bold(v)_i
  = lambda_1^k sum_(i = 1)^m accent(bold(b), tilde)_i bold(v)_i \

lim_(k -> oo) (bold(A)^k bold(b))/norm(bold(A)^k bold(b))_2
  = lim_(k -> oo) (lambda_1^k sum_(i = 1)^m accent(bold(b), tilde)_i bold(v)_i)/
    norm(lambda_1^k sum_(i = 1)^m accent(bold(b), tilde)_i bold(v)_i)_2
  = lim_(k -> oo) (sum_(i = 1)^m accent(bold(b), tilde)_i bold(v)_i)/
    norm(sum_(i = 1)^m accent(bold(b), tilde)_i bold(v)_i)_2
$ <limit> 

All that's left to do is to check for the eigenvector invariance:

$
bold(A)(lim_(k -> oo) (bold(A)^k bold(b)_0)/norm(bold(A)^k bold(b)_0)_2)
  = lim_(k -> oo) (sum_(i = 1)^m accent(bold(b), tilde)_i bold(A) bold(v)_i)/
    norm(sum_(i = 1)^m accent(bold(b), tilde)_i bold(v)_i)_2
$

Remember that $v_i (1 <= i <= m)$ are the eigenvectors corresponding to
$lambda_1$, so $bold(A)bold(v)_i = lambda_1 bold(v)_i$, therefore:

$
bold(A) (lim_(k -> oo) (bold(A)^k bold(b))/norm(bold(A)^k bold(b))_2)
  &= lim_(k -> oo) (sum_(i = 1)^m accent(bold(b), tilde)_i lambda_1 bold(v)_i)/
    norm(sum_(i = 1)^m accent(bold(b), tilde)_i bold(v)_i)_2
  = lambda_1 lim_(k -> oo) (sum_(i = 1)^m accent(bold(b), tilde)_i bold(v)_i)/
    norm(sum_(i = 1)^m accent(bold(b), tilde)_i bold(v)_i)_2 \
  &= lambda_1 (lim_(k -> oo) (bold(A)^k bold(b))/norm(bold(A)^k bold(b))_2) \
$

Which proves that the limit converges to an eigenvector corresponding the top
eigenvalue $lambda_1$. In the case of a unique top eigenvalue, the limit is
just the normalized top eigenvector:

$
lim_(k -> oo) (bold(A)^k bold(b))/norm(bold(A)^k bold(b))_2
  = (accent(bold(b), tilde)_1 bold(v)_1)/
    norm(accent(bold(b), tilde)_1 bold(v)_1)_2
  = bold(v)_1/norm(bold(v)_1)_2
$

Which is exactly the top eigenvector, and the algorithm doesn't depend on the
initial choice $bold(b)_0$. Otherwise, the direction of $bold(b)_0$ will affect
both the speed and result. If $bold(b)_0$ is selected such that
$accent(bold(b), tilde)_i = 0$ for all $i$ such that $abs(lambda_i) =
abs(lambda_1)$, the @limit is no longer true and the algorithm can only
converge to a smaller eigenvector. We also shown via @power-factor that this
algorithm is just computing a really large matrix power, and the recurrence
relation described in @power-recurrence is just a method for doing it while
avoiding overflow.

While the Power Iteration method is a simple algorithm with an interesting
connection to matrix power, its efficiency and utility are leave much to be
desired. The next sections go over some enhancements of the algorithm and ways
to generalize them to get more eigenvectors or even the full decomposition.

== Stopping condition and eigenvalue

Since the algorithm is just evaluating a limit as $k$ grows to indefinitely, to
implement we just have to pick a large enough $k$, which is the `n_iter`
parameter of the example implementation. But how large is "large enough"? The
convergence rate depends on the converge rate of $abs(lambda_i/lambda_1)^k$, which
is not constant among all matrices. So if we use a fixed number of iterations,
we'll get unreliable precision. To determine when to safely stop the algorithm,
we can use the eigenvalue equation $bold(A)bold(v) = lambda bold(v)$, but
replace $bold(v)$ and $lambda$ with the approximation $bold(b)$ and $mu$.
Rearranging, we get:

$
bold(b) mu = bold(A) bold(b)
$ <eigval-eqn>

If $bold(b)$ haven't converged to $bold(v)$, @eigval-eqn has no solution, but
we convert it into a least squares problem and minimize the residual:

$
(accent(bold(b), hat), accent(mu, hat)) = arg min_(bold(b), mu) norm(bold(b) mu - bold(A) bold(b))_2^2
$ <eigval-lstqr>

$bold(b)$ is already optimized by Power Iteration, so if we fix $bold(b)$ and
try to optimize for $mu$, we have:

$
accent(mu, hat)(bold(b))
  = arg min_mu norm(bold(b) mu - bold(A) bold(b))_2^2
  = (bold(b)^T bold(A) bold(b))/(bold(b)^T bold(b))
$ <rayleigh-quotient>

This is call the Rayleigh quotient of the vector $bold(b)$, or the best
approximation of the eigenvalue given an eigenvector approximation $bold(b)$.
We will come back to this later, but currently our objective is to find a
stopping condition. For this, we can use the error of @eigval-eqn.

$
norm(bold(b) mu - bold(A) bold(b))_2^2
  = norm(bold(b) (bold(b)^T bold(A) bold(b))/(bold(b)^T bold(b)) - bold(A)bold(b))_2^2
  = norm((bold(b) bold(b)^T bold(A) bold(b))/(bold(b)^T bold(b)) - bold(A)bold(b))_2^2
  = norm(((bold(b) bold(b)^T)/(bold(b)^T bold(b)) - I_n) bold(A)bold(b))_2^2
$

Since $b$ is normalized every iteration, you can drop the $bold(b)^T bold(b)$
term for simplification. Instead of running for a fixed number of iterations,
we can run until the error is below a fixed tolerance.

```python
def power_iteration(A: NPMat, rng: NPRng, tol=1e-12) -> NPVec:
  n, m = A.shape
  assert m == n, f"A ({m}×{n}) is non-square"

  I = np.eye(n)
  b: NPVec = rng.standard_normal((n,))
  b /= np.linalg.norm(b)

  while True:
    Ab = A @ b
    r = (np.outer(b) - I) @ Ab
    if r @ r < tol:
      return b

    b = Ab / np.linalg.norm(Ab) 
```

Most of the time parametrizing by tolerance is preferable because getting
accurate results is more important. In practice, you might want to throw an
error after a fixed number of iterations if it haven't converged. The choice of
the tolerance is also interesting I wont go into details here. More
importantly, we can also extend the algorithm to get the largest eigenvalue
using the Rayleigh quotient described in @rayleigh-quotient.

= Rayleigh quotient Iteration

But we can do more that just returning the eigenvalue. If we incorporate the
eigenvalue into the iteration process, and if we don't need the top
eigenvector, there's an algorithm with a much better convergence rate. It's
called Rayleigh quotient Iteration (RQI). And as its name suggest, it extend
power iteration with the Rayleigh quotient to find an eigenvector close to the
initial approximation. The convergence behavior of RQI is much more complicated
than plain Power Iteration, so the method won't get analyzed in detail. But
I'll show how to get from power iteration to RQI.

== Inverse Iteration

The first step is to instead find the eigenvector with the smallest absolute
eigenvalue. We can use the following result:

$
bold(A)^(-1) = mat(
  bar.v, bar.v, , bar.v;
  bold(v)_1, bold(v)_2, ..., bold(v)_n;
  bar.v, bar.v, , bar.v;
) mat(
  1/lambda_1, 0, ..., 0;
  0, 1/lambda_2, ..., 0;
  dots.v, dots.v, dots.down, dots.v;
  0, 0, ..., 1/lambda_n;
) mat(
  bar.v, bar.v, , bar.v;
  bold(v)_1, bold(v)_2, ..., bold(v)_n;
  bar.v, bar.v, , bar.v;
)^(-1)
$

You can easily verify this by multiplying it with $bold(A)$ or its EVD, which
indeeds result in the identity matrix. This allows us to extend the matrix
power formula to negative numbers by inverting and raising to the absolute
power. But more importantly, if $abs(lambda_1) >= abs(lambda_2) >= ... >=
abs(lambda_n)$, then $abs(1/lambda_1) <= abs(1/lambda_2) <= ... <=
abs(1/lambda_n)$, in other words, the inverse of a matrix $bold(A)^(-1)$ has
the same eigenvectors of the original matrix, but eigenvalues have their order
of absolute values reversed. So to find the smallest eigenvector, we can apply
Power iteration to $bold(A)^(-1)$, or:

$
bold(b)_(k + 1) = (bold(A)^(-1) bold(b)_k)/norm(bold(A)^(-1) bold(b)_k)_2
$

Note that if $bold(A)$ is singular, then the smallest eigenvalue is $0$, and
the eigenvector is in the nullspace of $bold(A)$ which needs to be computed
with a different method. Also, if the smallest eigenvalue is near $0$,
inverting $bold(A)$ will be extremely unstable. So instead of inverting, we can
instead solve a linaer system of equation:

$
bold(A) bold(b)_(k + 1) = bold(b)_k
$

Then we can normalize the result to avoid overflowing. The system can be solved
by numerically stable methods, such as LU Decomposition in the general case,
Bunch-Kaufman factorization if the matrix is symmetric, and Cholesky
decomposition if the matrix is positive-definite. The most expensive part of
these method is the matrix decomposition, but it only have to be done once at
the start of the algorithm.

== The Rayleigh quotient method

Once we can find the smallest eigenvector, we can easily get the eigenvector
close to an initial eigenvector. Instead of minimizing $abs(lambda)$, we can
instead minimize $abs(lambda - mu)$, where $mu$ is our target eigenvalue. To
subtract $mu$ from each eigenvalue while retain the eigenvectors, we can use
this matrix:

$
mat(
  bar.v, bar.v, , bar.v;
  bold(v)_1, bold(v)_2, ..., bold(v)_n;
  bar.v, bar.v, , bar.v;
) mat(
  lambda_1 - mu, 0, ..., 0;
  0, lambda_2 - mu, ..., 0;
  dots.v, dots.v, dots.down, dots.v;
  0, 0, ..., lambda_n - mu;
) mat(
  bar.v, bar.v, , bar.v;
  bold(v)_1, bold(v)_2, ..., bold(v)_n;
  bar.v, bar.v, , bar.v;
)^(-1)
$

Which you can work out to be $bold(A) - mu bold(I)_n$. To find the smallest
eigenvector of this matrix, we can apply Inverse Iteration as described in the
previous section.

But what value of $mu$ should we use, the closer $mu$ is to an eigenvalue
$lambda$, the faster the algorithm will converge as $abs(1/(mu - lambda))$
quickly dominate. And how to get an eigenvalue approximation from an
eigenvector? The Rayleigh quotient. Therefore, the update step of RQI is as
follow:

$
bold(b)_(k + 1) =
  ((bold(A) - I_n bold(b)_k^T bold(A) bold(b)_k)^(-1) bold(b)_k)/
  norm((bold(A) - I_n bold(b)_k^T bold(A) bold(b)_k)^(-1) bold(b)_k)_2
$

Again, use a numerically stable system solver instead of inverting, you can
even decompose $A$ once at the start and perform a rank-1 update each
iteration. This algorithm is very fast, but since each iteration is expensive,
you may want to initialize it with a good approximation of $b_0$. This is why
RQI is often used as an "Eigenpair refinement algorithm", where you have a
crude approximation for an eigenvector/eigenvalue pair, and want to reduce the
approximation error. Here's an implementation that requires at least an initial
eigenvector:

```python
def rayleigh_quotient_iteration(
  A: NPMat,
  b: NPVec,
  mu: float | None = None,
  tol=1e-12,
) -> tuple[float, NPVec]:
  n, m = A.shape
  assert m == n, f"A ({m}×{n}) is non-square"

  I = np.eye(n)
  b /= np.linalg.norm(b)
  mu = b.T @ A @ b if mu is None else mu

  while True:
    try:
      w = np.linalg.solve(A - I * mu)
    except:
      return mu, b
    
    b = w / np.linalg.norm()
    Ab = A @ b
    mu = b.T @ Ab
    r = mu * b - Ab

    if r @ r < tol:
      return mu, b
```

= Deterministic Power Iteration

RQI is a good choice for improving the precision of eigenvector/eigenvalue
pairs, but when you need the top eigenvector, such as when finding the
direction of maximum variance, you either have to compute all of them or go
back to Power Iteration. Which means that you have to give up the fast
convergence rate of RQI.

Another problem that plagues both method is the sensivity to initialization.
There's nothing wrong with randomized algorithms, but it's interesting to see
if we can make Power Iteration work without having to select the initial
vector. Just a head up about practicality, what I'm about to describe here
takes away one big advantage of power iteration: its ability to work
efficiently with sparse and black-box matrices.

In practice, if the matrix is dense, you're better of using more advanced
methods to compute all eigenvectors and pick the one with the largest
eigenvalue, since these algorithm converges much faster anyways. But an
advantage of this method is that it's reayy simple to implement, which may or
may not justify its usage.

== The limit of normalized matrix power

The idea is to remove $b_0$ from power iteration formula, and just see what it
converges to:

$
lim_(k -> oo) bold(A)^k/norm(bold(A)^k)_F
$

Where $norm(bold(A))_F$ is the Frobenius norm, which can be expressed as:

$
norm(bold(A))_F
  = sqrt(tr(bold(A)^T bold(A)))
  = sqrt(lambda_1^2 + lambda_2^2 + ... + lambda_n^2)
$

Evaluating the limit similar to @power-proof gives us:

$
lim_(k -> oo) bold(A)^k
  = lambda_1^k sum_(i=1)^m bold(v)_i bold(w)_i \
lim_(k -> oo) norm(bold(A)^k)_F
  = lambda_1^k sqrt(m) \
lim_(k -> oo) bold(A)^k/norm(bold(A)^k)_F
  = 1/sqrt(m) sum_(i=1)^m bold(v)_i bold(w)_i
$

Where $(bold(w)_1^T, bold(w)_2^T, ..., bold(w)_n^T) in RR^(n times n)$ are the
rows of $P^(-1)$, in other words:

$
bold(A) = bold(P) bold(D) bold(P)^(-1) = mat(
  bar.v, bar.v, , bar.v;
  bold(v)_1, bold(v)_2, ..., bold(v)_n;
  bar.v, bar.v, , bar.v;
) mat(
  lambda_1, 0, ..., 0;
  0, lambda_2, ..., 0;
  dots.v, dots.v, dots.down, dots.v;
  0, 0, ..., lambda_n;
) mat(
  bar.h, bold(w)_1, bar.h;
  bar.h, bold(w)_2, bar.h;
  , dots.v, ;
  bar.h, bold(w)_2, bar.h;
)
$

We can rewrite the result of the limit in Matrix form:

$
lim_(k -> oo) bold(A)^k/norm(bold(A)^k)_F
  = mat(
  bar.v, , bar.v, , bar.v;
  bold(v)_1, ..., bold(v)_m, ..., bold(v)_n;
  bar.v, , bar.v, , bar.v;
) mat(
  1/sqrt(m), ..., 0, ..., 0;
  dots.v, dots.down, dots.v, , dots.v;
  0, ..., 1/sqrt(m), ..., 0;
  dots.v, , dots.v, dots.down, dots.v;
  0, ..., 0, ..., 0;
) mat(
  bar.h, bold(w)_1, bar.h;
  , dots.v, ;
  bar.h, bold(w)_m, bar.h;
  , dots.v, ;
  bar.h, bold(w)_n, bar.h;
)
$

Then take the matrix-vector product of the limit and a vector $bold(b)$, we
get:

$
(lim_(k -> oo) bold(A)^k/norm(bold(A)^k)_F)bold(b)
  &= (lim_(k -> oo) bold(A)^k/norm(bold(A)^k)_F) bold(P) accent(bold(b), tilde) \
  &= mat(
  bar.v, , bar.v, , bar.v;
  bold(v)_1, ..., bold(v)_m, ..., bold(v)_n;
  bar.v, , bar.v, , bar.v;
) mat(
  1/sqrt(m), ..., 0, ..., 0;
  dots.v, dots.down, dots.v, , dots.v;
  0, ..., 1/sqrt(m), ..., 0;
  dots.v, , dots.v, dots.down, dots.v;
  0, ..., 0, ..., 0;
) vec(
  accent(bold(b), tilde)_1,
  dots.v,
  accent(bold(b), tilde)_m,
  dots.v,
  accent(bold(b), tilde)_n,
) \
  &= 1/sqrt(m) sum_(i = 0)^m bold(v)_i accent(bold(b), tilde)_i
$

So this means that we get the same result as plain Power Iteration, but instead
of selecting a vector at the start and hope that it's not orthogonal to the
largest eigenvector, we can evaluate the limit *then* compute the vector
product. If we encounter an orthogonal vector then we can just discard it and
try another vector. We can't do this with plain Power Iteration because
everytime we try another vector, we have to restart the entire iteration
process. 

Moreover, we can try trivial vectors, such as the standard basis vectors $(e_1,
e_2, ..., e_n)$ and just pick any non-zero product. If we exhaust all standard
basis vector and can't find a non-zero product then we can safely confirm that
$bold(A)$ is the zero matrix and any vector is the largest eigenvector.
Multiplying a matrix with a standard basis vector $e_i$ is essentially
extracting the $i$th column of the matrix, so we can return the column with the
largest norm as a robust non-zero eigenvector. This gives us a deterministic
implementation as follow:

```python
def matrix_power_iteration(A: NPMat, n_iter: int) -> NPVec:
  n, m = A.shape
  assert m == n, f"A ({m}×{n}) is non-square"

  norm2 = (A * A).sum()
  if norm2 < np.finfo(A.dtype).eps:
    v = np.zeros_like(A[0])
    v[0] = 1
    return v

  B = A.copy()
  for _ in range(n_iter):
    B /= np.linalg.norm(B)
    B = A @ B

  col_norm = np.linalg.norm(B, axis=0)
  max_col = col_norm.argmax()
  return B[:, max_col] / col_norm[max_col]
```

Again, there's the concern about stopping condition, and for that you can use
the squared norm of the difference between consecutive iterations. But instead
of that I'm going to propose a simple modification to the algorithm that make
it so fast that you don't need a stopping condition.

== Exponentiation by squaring

Back in @practical-consideration, I mentioned how to compute $A^k$, we
can use a method called exponentation by squaring, and `np.linalg.matrix_power`
implements it. It turns out that you can easily use it decrease the number of
iterations. We can rewrite the iteration step:

$
bold(B)_k &= bold(A)^k/norm(bold(A)^k)_F \
bold(B)_(2k) &= (bold(A)^k bold(A)^k)/norm(bold(A)^k bold(A)^k)_F
  = (bold(A)^k/norm(bold(A)^k)_F bold(A)^k/norm(bold(A)^k)_F)
    /norm(bold(A)^k/norm(bold(A)^k)_F bold(A)^k/norm(bold(A)^k)_F)_F \
  &= (bold(B)_k bold(B)_k)/norm(bold(B)_k bold(B)_k)_F
  = (bold(B)_k^2)/norm(bold(B)_k^2)_F
$

Therefore instead of multiplying by $bold(A)$ each step, we can just square and
renormalize. This basically reduce the number of iterations exponentially, so
if previously we need 1 million iterations, now we only need about 20. And the
cost per iteration is the same as the previous method as we just change what
matrix to multiply by each step.

```python
def squaring_power_iteration(A: NPMat) -> NPVec:
  n, m = A.shape
  assert m == n, f"A ({m}×{n}) is non-square"

  norm2 = (A * A).sum()
  if norm2 < np.finfo(A.dtype).eps:
    v = np.zeros_like(A[0])
    v[0] = 1
    return v

  for _ in range(24):
    A /= np.linalg.norm(A)
    A = A @ A

  col_norm = np.linalg.norm(A, axis=0)
  max_col = col_norm.argmax()
  return A[:, max_col] / col_norm[max_col]
```

In the code, I fixed the number of iterations to 24, because that's more than
enough for practical purposes. This gives us a simple, fast, and deterministic
algorithm for finding the top eigenvector, but since matrix-matrix
multiplication is $cal(O)(n^3)$, this should only be used for small matrices,
and when you only care about the top eigenvector. Otherwise, switch back to
plain power iteration or find all eigenvectors with a different method.

Another important thing to note is that it's not necessary to use the Frobenius
norm for each iteration step. You can use other norm that's easier to compute
such as the maximum component norm $display(max_(i, j) abs(A_(i, j)))$ and it
will still converge as long as the submultiplicativity property is preserved.
In case you really want the algorithm to fully converge regardless of the input
matrix, set the number of iteration to the number of bits of the machine
precision.

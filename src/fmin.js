/// searches along line 'pk' for a point that satifies the wolfe conditions
/// See 'Numerical Optimization' by Nocedal and Wright p59-60
/// f : objective function
/// pk : search direction
/// current: object containing current gradient/loss
/// next: output: contains next gradient/loss
/// returns a: step size taken
export function wolfeLineSearch(f, pk, current, next, a, c1, c2) {
    const phi0 = current.fx;
    const phiPrime0 = dot(current.fxprime, pk);
    let phi = phi0;
    let phi_old = phi0;
    let phiPrime = phiPrime0;
    let a0 = 0;

    a = a || 1;
    c1 = c1 || 1e-6;
    c2 = c2 || 0.1;

    function zoom(a_lo, a_high, phi_lo) {
        for (let iteration = 0; iteration < 16; ++iteration) {
            a = (a_lo + a_high) / 2;
            weightedSum(next.x, 1.0, current.x, a, pk);
            phi = next.fx = f(next.x, next.fxprime);
            phiPrime = dot(next.fxprime, pk);

            if (phi > phi0 + c1 * a * phiPrime0 || phi >= phi_lo) {
                a_high = a;
            } else {
                if (Math.abs(phiPrime) <= -c2 * phiPrime0) {
                    return a;
                }

                if (phiPrime * (a_high - a_lo) >= 0) {
                    a_high = a_lo;
                }

                a_lo = a;
                phi_lo = phi;
            }
        }

        return 0;
    }

    for (let iteration = 0; iteration < 10; ++iteration) {
        weightedSum(next.x, 1.0, current.x, a, pk);
        phi = next.fx = f(next.x, next.fxprime);
        phiPrime = dot(next.fxprime, pk);
        if (phi > phi0 + c1 * a * phiPrime0 || (iteration && phi >= phi_old)) {
            return zoom(a0, a, phi_old);
        }

        if (Math.abs(phiPrime) <= -c2 * phiPrime0) {
            return a;
        }

        if (phiPrime >= 0) {
            return zoom(a, a0, phi);
        }

        phi_old = phi;
        a0 = a;
        a *= 2;
    }

    return a;
}

export function conjugateGradient(f, initial, params) {
    // allocate all memory up front here, keep out of the loop for perfomance
    // reasons
    let current = { x: initial.slice(), fx: 0, fxprime: initial.slice() };
    let next = { x: initial.slice(), fx: 0, fxprime: initial.slice() };
    const yk = initial.slice();
    let pk;
    let temp;
    let a = 1;
    let maxIterations;

    params = params || {};
    maxIterations = params.maxIterations || initial.length * 20;

    current.fx = f(current.x, current.fxprime);
    pk = current.fxprime.slice();
    scale(pk, current.fxprime, -1);

    for (let i = 0; i < maxIterations; ++i) {
        a = wolfeLineSearch(f, pk, current, next, a);

        // todo: history in wrong spot?
        if (params.history) {
            params.history.push({
                x: current.x.slice(),
                fx: current.fx,
                fxprime: current.fxprime.slice(),
                alpha: a,
            });
        }

        if (!a) {
            // faiiled to find point that satifies wolfe conditions.
            // reset direction for next iteration
            scale(pk, current.fxprime, -1);
        } else {
            // update direction using Polak–Ribiere CG method
            weightedSum(yk, 1, next.fxprime, -1, current.fxprime);

            const delta_k = dot(current.fxprime, current.fxprime);
            const beta_k = Math.max(0, dot(yk, next.fxprime) / delta_k);

            weightedSum(pk, beta_k, pk, -1, next.fxprime);

            temp = current;
            current = next;
            next = temp;
        }

        if (norm2(current.fxprime) <= 1e-5) {
            break;
        }
    }

    if (params.history) {
        params.history.push({
            x: current.x.slice(),
            fx: current.fx,
            fxprime: current.fxprime.slice(),
            alpha: a,
        });
    }

    return current;
}

/// Solves a system of lienar equations Ax =b for x
/// using the conjugate gradient method.
export function conjugateGradientSolve(A, b, x, history) {
    const r = x.slice();
    const Ap = x.slice();
    let p;
    let rsold;
    let rsnew;
    let alpha;

    // r = b - A*x
    gemv(Ap, A, x);
    weightedSum(r, 1, b, -1, Ap);
    p = r.slice();
    rsold = dot(r, r);

    for (let i = 0; i < b.length; ++i) {
        gemv(Ap, A, p);
        alpha = rsold / dot(p, Ap);
        if (history) {
            history.push({ x: x.slice(), p: p.slice(), alpha: alpha });
        }

        //x=x+alpha*p;
        weightedSum(x, 1, x, alpha, p);

        // r=r-alpha*Ap;
        weightedSum(r, 1, r, -alpha, Ap);
        rsnew = dot(r, r);
        if (Math.sqrt(rsnew) <= 1e-10) break;

        // p=r+(rsnew/rsold)*p;
        weightedSum(p, 1, r, rsnew / rsold, p);
        rsold = rsnew;
    }
    if (history) {
        history.push({ x: x.slice(), p: p.slice(), alpha: alpha });
    }
    return x;
}
/** finds the zeros of a function, given two starting points (which must
 * have opposite signs */
export function bisect(f, a, b, parameters) {
    parameters = parameters || {};
    const maxIterations = parameters.maxIterations || 100;
    const tolerance = parameters.tolerance || 1e-10;
    const fA = f(a);
    const fB = f(b);
    let delta = b - a;

    if (fA * fB > 0) {
        throw 'Initial bisect points must have opposite signs';
    }

    if (fA === 0) return a;
    if (fB === 0) return b;

    for (let i = 0; i < maxIterations; ++i) {
        delta /= 2;
        const mid = a + delta;
        const fMid = f(mid);

        if (fMid * fA >= 0) {
            a = mid;
        }

        if (Math.abs(delta) < tolerance || fMid === 0) {
            return mid;
        }
    }
    return a + delta;
}

// need some basic operations on vectors, rather than adding a dependency,
// just define here
export function zeros(x) {
    const r = new Array(x);
    for (let i = 0; i < x; ++i) {
        r[i] = 0;
    }
    return r;
}
export function zerosM(x, y) {
    return zeros(x).map(() => zeros(y));
}

export function dot(a, b) {
    let ret = 0;
    for (let i = 0; i < a.length; ++i) {
        ret += a[i] * b[i];
    }
    return ret;
}

export function norm2(a) {
    return Math.sqrt(dot(a, a));
}

export function scale(ret, value, c) {
    for (let i = 0; i < value.length; ++i) {
        ret[i] = value[i] * c;
    }
}

export function weightedSum(ret, w1, v1, w2, v2) {
    for (let j = 0; j < ret.length; ++j) {
        ret[j] = w1 * v1[j] + w2 * v2[j];
    }
}

export function gemv(output, A, x) {
    for (let i = 0; i < output.length; ++i) {
        output[i] = dot(A[i], x);
    }
}
/** minimizes a function using the downhill simplex method */
export function nelderMead(f, x0, parameters) {
    parameters = parameters || {};

    const maxIterations = parameters.maxIterations || x0.length * 200;
    const nonZeroDelta = parameters.nonZeroDelta || 1.05;
    const zeroDelta = parameters.zeroDelta || 0.001;
    const minErrorDelta = parameters.minErrorDelta || 1e-6;
    const minTolerance = parameters.minErrorDelta || 1e-5;
    const rho = parameters.rho !== undefined ? parameters.rho : 1;
    const chi = parameters.chi !== undefined ? parameters.chi : 2;
    const psi = parameters.psi !== undefined ? parameters.psi : -0.5;
    const sigma = parameters.sigma !== undefined ? parameters.sigma : 0.5;
    let maxDiff;

    // initialize simplex.
    const N = x0.length;
    const simplex = new Array(N + 1);
    simplex[0] = x0;
    simplex[0].fx = f(x0);
    simplex[0].id = 0;
    for (let i = 0; i < N; ++i) {
        const point = x0.slice();
        point[i] = point[i] ? point[i] * nonZeroDelta : zeroDelta;
        simplex[i + 1] = point;
        simplex[i + 1].fx = f(point);
        simplex[i + 1].id = i + 1;
    }

    function updateSimplex(value) {
        for (let i = 0; i < value.length; i++) {
            simplex[N][i] = value[i];
        }
        simplex[N].fx = value.fx;
    }

    const sortOrder = (a, b) => a.fx - b.fx;

    const centroid = x0.slice();
    const reflected = x0.slice();
    const contracted = x0.slice();
    const expanded = x0.slice();

    for (let iteration = 0; iteration < maxIterations; ++iteration) {
        simplex.sort(sortOrder);

        if (parameters.history) {
            // copy the simplex (since later iterations will mutate) and
            // sort it to have a consistent order between iterations
            const sortedSimplex = simplex.map((x) => {
                const state = x.slice();
                state.fx = x.fx;
                state.id = x.id;
                return state;
            });
            sortedSimplex.sort((a, b) => a.id - b.id);

            parameters.history.push({
                x: simplex[0].slice(),
                fx: simplex[0].fx,
                simplex: sortedSimplex,
            });
        }

        maxDiff = 0;
        for (let i = 0; i < N; ++i) {
            maxDiff = Math.max(maxDiff, Math.abs(simplex[0][i] - simplex[1][i]));
        }

        if (Math.abs(simplex[0].fx - simplex[N].fx) < minErrorDelta && maxDiff < minTolerance) {
            break;
        }

        // compute the centroid of all but the worst point in the simplex
        for (let i = 0; i < N; ++i) {
            centroid[i] = 0;
            for (let j = 0; j < N; ++j) {
                centroid[i] += simplex[j][i];
            }
            centroid[i] /= N;
        }

        // reflect the worst point past the centroid  and compute loss at reflected
        // point
        const worst = simplex[N];
        weightedSum(reflected, 1 + rho, centroid, -rho, worst);
        reflected.fx = f(reflected);

        // if the reflected point is the best seen, then possibly expand
        if (reflected.fx < simplex[0].fx) {
            weightedSum(expanded, 1 + chi, centroid, -chi, worst);
            expanded.fx = f(expanded);
            if (expanded.fx < reflected.fx) {
                updateSimplex(expanded);
            } else {
                updateSimplex(reflected);
            }
        }

        // if the reflected point is worse than the second worst, we need to
        // contract
        else if (reflected.fx >= simplex[N - 1].fx) {
            let shouldReduce = false;

            if (reflected.fx > worst.fx) {
                // do an inside contraction
                weightedSum(contracted, 1 + psi, centroid, -psi, worst);
                contracted.fx = f(contracted);
                if (contracted.fx < worst.fx) {
                    updateSimplex(contracted);
                } else {
                    shouldReduce = true;
                }
            } else {
                // do an outside contraction
                weightedSum(contracted, 1 - psi * rho, centroid, psi * rho, worst);
                contracted.fx = f(contracted);
                if (contracted.fx < reflected.fx) {
                    updateSimplex(contracted);
                } else {
                    shouldReduce = true;
                }
            }

            if (shouldReduce) {
                // if we don't contract here, we're done
                if (sigma >= 1) break;

                // do a reduction
                for (let i = 1; i < simplex.length; ++i) {
                    weightedSum(simplex[i], 1 - sigma, simplex[0], sigma, simplex[i]);
                    simplex[i].fx = f(simplex[i]);
                }
            }
        } else {
            updateSimplex(reflected);
        }
    }

    simplex.sort(sortOrder);
    return { fx: simplex[0].fx, x: simplex[0] };
}
import numpy as np


def approx_fprime(f, x, d_out, eps=np.finfo(np.float32).eps):
    """Numerically approximate the gradient of f at point x using central differences."""
    grad = np.zeros_like(x)
    mask = np.zeros_like(x)
    for ix in np.ndindex(x.shape):
        mask[ix] = eps
        fplus = f(x + mask)
        fminus = f(x - mask)
        mask[ix] = 0.0
        grad[ix] = np.sum(d_out * (fplus - fminus)) / (2 * eps)
    return grad


def check_affine(affine):
    # Check shapes: forward
    X = np.random.normal(size=(33, 11))
    W = np.random.normal(size=(11, 17))
    b = np.random.normal(size=(17))
    out = affine.forward(X, W, b)
    if out.shape != (33, 17):
        raise AssertionError(f"Shape of output {out.shape} doesn't match the expected shape (33, 17).")

    # Check shapes: backward
    d_X, d_W, d_b = affine.backward(np.ones(shape=(33,17)))
    if d_X.shape != X.shape:
        raise AssertionError(f"Shape of d_X {d_X.shape} doesn't match the shape of X {X.shape}.")
    if d_W.shape != W.shape:
        raise AssertionError(f"Shape of d_W {d_W.shape} doesn't match the shape of W {W.shape}.")
    if d_b.shape != b.shape:
        raise AssertionError(f"Shape of d_b {d_b.shape} doesn't match the shape of b {b.shape}.")

    # Check forward pass
    X = np.array([[1., 0.2, -0.1],
                  [1., 4., -1.],
                  [2., 2., -0.5],
                  [1., 0.3, -1.]])
    W = np.array([[5., 4.],
                  [-2., 1],
                  [3., 0.5]])
    b = np.array([5., 4.])

    out_correct = np.array([[ 9.3 ,  8.15],
                            [-1.  , 11.5 ],
                            [ 9.5 , 13.75],
                            [ 6.4 ,  7.8 ]])
    out = affine.forward(X, W, b)

    if not np.allclose(out, out_correct):
        raise AssertionError(f"Forward doesn't produce the expected result.")

    # Check backward pass
    X = np.random.normal(size=(33, 11))
    W = np.random.normal(size=(11, 17))
    b = np.random.normal(size=(17))
    d_out = np.random.normal(size=(33, 17))

    _ = affine.forward(X, W, b)
    d_X, d_W, d_b = affine.backward(d_out)

    f_X = lambda X: affine.forward(X, W, b)
    d_X_num = approx_fprime(f_X, X, d_out)

    if not np.allclose(d_X, d_X_num):
        raise AssertionError(f"Differnce between numerical and analytical gradients d_X is significant.")

    f_W = lambda W: affine.forward(X, W, b)
    d_W_num = approx_fprime(f_W, W, d_out)
    if not np.allclose(d_W, d_W_num):
        raise AssertionError(f"Differnce between numerical and analytical gradients d_W is significant.")

    f_b = lambda b: affine.forward(X, W, b)
    d_b_num = approx_fprime(f_b, b, d_out)
    if not np.allclose(d_b, d_b_num):
        raise AssertionError(f"Differnce between numerical and analytical gradients d_b is significant.")

    print('All checks passed succesfully!')


def check_relu(relu):
    # Check shapes: forward
    x = np.random.normal(size=(17, 13))
    out = relu.forward(x)
    if x.shape != out.shape:
        raise AssertionError(f"Shape of output {out.shape} doesn't match the shape of inputs {x.shape}.")

    # Check shapes: backward
    d_x = relu.backward(np.ones_like(out))
    if d_x.shape != x.shape:
        raise AssertionError(f"Shape of d_inputs {d_x.shape} doesn't match the shape of inputs {x.shape}.")

    # Check forward pass
    x = np.array([[1., 2., -3.],
                  [-0.5, 0.2, 14.]])
    out_correct = np.array([[1., 2., 0.],
                            [0., 0.2, 14.]])
    out = relu.forward(x)
    if not np.allclose(out, out_correct):
        raise AssertionError(f"Forward doesn't produce the expected result.")

    # Check backward pass
    x = np.random.normal(size=(17, 13))
    d_out = np.random.normal(size=(17, 13))

    f = lambda x: relu.forward(x)
    d_x_num = approx_fprime(f, x, d_out)

    _ = relu.forward(x)
    d_x = relu.backward(d_out)
    if not np.allclose(d_x, d_x_num):
        raise AssertionError(f"Differnce between numerical and analytical gradients is significant.")

    print('All checks passed succesfully!')


def check_cross_entropy(cross_entropy):
    # Checking forward shape is already implemented inside the function

    # Check shapes: backward
    logits = np.random.normal(size=(41, 21))
    y = np.random.randint(low=0, high=21, size=[41])
    labels = np.zeros(shape=(41, 21))
    labels[np.arange(41), y] = 1.0

    out = cross_entropy.forward(logits, labels)
    d_logits, d_labels = cross_entropy.backward(1.0)
    if not d_logits.shape == (41, 21):
        raise AssertionError(f"Shape of d_logits {d_logits.shape} doesn't match the shape of logits {logits.shape}.")

    # Check forward pass
    logits = np.array([[ 9.3 ,  8.15],
                       [-1.  , 11.5 ],
                       [ 9.5 , 13.75],
                       [ 6.4 ,  7.8 ]])
    labels = np.array([[1., 0.],
                       [1., 0.],
                       [0., 1.],
                       [1., 0.]])
    out_correct = 3.6024162941706455
    out = cross_entropy.forward(logits, labels)

    if not np.allclose(out, out_correct):
        raise AssertionError(f"Forward doesn't produce the expected result.")

    # Check backward pass
    logits = np.random.normal(size=(41, 21))
    y = np.random.randint(low=0, high=21, size=[41])
    labels = np.zeros(shape=(41, 21))
    labels[np.arange(41), y] = 1.0
    d_out = np.random.normal()

    _ = cross_entropy.forward(logits, labels)
    d_logits, _ = cross_entropy.backward(d_out)

    f_logits = lambda x: cross_entropy.forward(x, labels)
    d_logits_num = approx_fprime(f_logits, logits, d_out)

    if not np.allclose(d_logits, d_logits_num):
        raise AssertionError(f"Differnce between numerical and analytical gradients is significant.")

    print('All checks passed succesfully!')

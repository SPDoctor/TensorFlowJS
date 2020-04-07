import tf from "tensorflow"

// Convert a tensor of form `[H,W,C,CO]` to form `[1,H',W',CO] where the `C`
// rank is turned into a grid of objects to visualise activation layers of images.
function gridify(input, pad, I, J) {
    let x = tf.pad(input, tf.constant([[pad,pad],[pad,pad],[0,0],[0,0]]))
    let [H,W,C,CO] = input.get_shape();
    H = H + 2 * pad;
    W = W + 2 * pad;
    if (J === undefined) {
        J = Math.floor(Math.sqrt(C));
        while (C % J !== 0) {
            J--;
        }
        I = Math.floor(C / J);
    }
    if (I * J !== C) {
        throw new Error("oops my code doesn't make a grid.")
    }
    x = tf.transpose(x, [2,0,1,3]);
    x = tf.reshape(x, [J,I * H, W, CO]);
    x = tf.transpose(x, [0,2,1,3])
    x = tf.reshape(x, [1, J * W, I * H, CO])
    x = tf.transpose(x,[0,2,1,3])
    return x  //: (1, H * I, W * J, CO)
}
/*
// original python code:
def gridify(input, pad=2, I=None, J=None): # input : (H,W,C,CO)
    """Takes as input a tensor of shape `(H,W,C,CO)` and returns a tensor of shape `(1, (H+2*pad) * I, (W+2*pad) * J, CO)`.
    Use it to visualise activation layers of images.
    If the input tensor has rank 3 it is extended so that `CO=1`.
    If `I` and `J` are not given they are calculated as the integer factorisation `C = I * J` closest to a square.
    """
    if len(input.get_shape()) == 3: input = tf.expand_dims(input, axis=-1)
    x = tf.pad(input, tf.constant([[pad,pad],[pad,pad],[0,0],[0,0]]))
    H,W,C,CO = input.get_shape()
    H = H + 2 * pad
    W = W + 2 * pad
    if J is None:
        J = int(math.sqrt(int(C)))
        while (C % J != 0):
            J -= 1
        I = C // J
    assert I * J == C
    x = tf.transpose(x, (2,0,1,3))
    x = tf.reshape(x, (J, I * H, W, CO))
    x = tf.transpose(x, (0,2,1,3))
    x = tf.reshape(x, (1, J * W, I * H, CO))
    x = tf.transpose(x,(0,2,1,3))
    return x # : (1, H * I, W * J, CO)
 */
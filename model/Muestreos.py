import tensorflow as tf


def A_Fran(y, Masks):
    mul_meas = tf.multiply(tf.math.conj(Masks), y)
    return  tf.signal.fft2d(mul_meas)

def AT_Fran(y, Masks):
    mult_mass_z = tf.multiply(Masks, tf.signal.ifft2d(y))
    res = tf.reduce_sum(mult_mass_z, axis=1, keepdims=True)
    return tf.multiply(res, tf.cast(y.shape[2]*y.shape[3], dtype=res.dtype))


def A_ASM_LAB(y, Masks, SFTF, kappa):
    y = tf.signal.fftshift(tf.signal.fft2d(tf.signal.fftshift(tf.multiply(y, Masks))))
    y = tf.signal.ifftshift(tf.signal.ifft2d(tf.signal.fftshift(tf.multiply(y, SFTF))))
    y = tf.square(tf.math.abs(y))
    res = tf.math.divide(tf.math.sqrt(y), tf.math.sqrt(kappa))
    return  res

def AT_ASM_LAB(y, Masks, SFTF):
      y = tf.signal.fftshift(y, axes=[2,3]);
      y = tf.signal.fftshift(tf.signal.fft2d(y), axes=[2,3]);
      y = tf.signal.fftshift(tf.multiply(SFTF,y), axes=[2,3]);
      y = tf.signal.ifftshift(tf.signal.ifft2d(y), axes=[2,3]);
      res = tf.reduce_mean(tf.multiply(tf.math.conj(Masks), y), axis=1)
      return res

def A_FRESNELL(x, Masks,Q, kappa):
    Q1, Q2 = Q
    x_q1 = tf.multiply(x, Q1)
    x_q1_mask = tf.signal.fft2d(tf.multiply(x_q1, tf.math.conj(Masks)))
    res = tf.multiply(x_q1_mask, Q2)*x.shape[0]*x.shape[1]
    res = tf.math.divide(tf.math.sqrt(tf.square(tf.math.abs(res))), tf.math.sqrt(kappa))
    return  res
    
def AT_FRESNELL(x, Masks,Q):
    Q1, Q2 = Q
    x_q2 = tf.signal.ifft2d(tf.multiply(x, tf.math.conj(Q2)))
    x_q1_q2 = tf.multiply(x_q2, tf.math.conj(Q1))
    x_q12_mask = tf.multiply(x_q1_q2, Masks)
    res = tf.reduce_mean(x_q12_mask, axis=1)
    return  res


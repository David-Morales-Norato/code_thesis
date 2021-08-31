import tensorflow as tf


def A_Fran(y, Masks):
    kappa=tf.constant(1e-3, tf.float64)
    print("Y - A SHAPE", y.shape, y.dtype)
    print("Masks - A SHAPE", Masks.shape, Masks.dtype)
    mul_meas = tf.multiply(tf.math.conj(Masks), y)
    y = tf.signal.fft2d(mul_meas)
    y = tf.cast(tf.square(tf.math.abs(y)), dtype=tf.float64)
    
    y = tf.math.divide(tf.math.sqrt(y), tf.math.sqrt(kappa))
    return  y

def AT_Fran(y, Masks):
    print("Y - A SHAPE", y.shape, y.dtype)
    print("Masks - A SHAPE", Masks.shape, Masks.dtype)
    mult_mass_z = tf.multiply(Masks, tf.signal.ifft2d(y))
    print("Y - A SHAPE", y.shape, y.dtype)
    print("mult_mass_z - A SHAPE", mult_mass_z.shape, mult_mass_z.dtype)
    res = tf.reduce_sum(mult_mass_z, axis=1, keepdims=True)
    return tf.multiply(res, tf.cast(y.shape[2]*y.shape[3], dtype=res.dtype))


# def A_ASM(y, Masks, SFTF):

#     mask_y = tf.signal.ifft2d(tf.multiply(y, tf.math.conj(Masks)))

#     mask_y_sftf = tf.signal.fft2d(tf.multiply(mask_y, SFTF))
#     res = tf.multiply(tf.math.conj(Masks), y)
#     return  tf.multiply(res, tf.math.sqrt(tf.cast(y.shape[2]*y.shape[3], dtype=res.dtype)))

# def AT_ASM(y, Masks, SFTF):
#     y_sftf = tf.signal.fft2d(tf.multiply(tf.signal.ifft2d(y), tf.math.conj(SFTF)))
#     mult_mass_z = tf.multiply(Masks, y_sftf)
#     res = tf.reduce_sum(mult_mass_z, axis=1, keepdims=True)
#     return tf.multiply(res, tf.math.sqrt(tf.cast(y.shape[2]*y.shape[3], dtype=res.dtype)))


def A_ASM_LAB(y, Masks, SFTF, kappa=1e-3):
    # print("Y - A SHAPE", y.shape, y.dtype)
    # print("Masks - A SHAPE", Masks.shape, Masks.dtype)
    # print("SFTF - A SHAPE", SFTF.shape, SFTF.dtype)
    y1 = tf.signal.fftshift(tf.multiply(y, Masks) , axes=[2,3])
    # print("SFTF - A SHAPE", SFTF.shape, SFTF.dtype)
    y2 = tf.signal.fftshift(tf.signal.fft2d(y1), axes=[2,3])
    y3 = tf.signal.fftshift(tf.multiply(y2, SFTF), axes=[2,3])
    y4 = tf.signal.ifftshift(tf.signal.ifft2d(y3), axes=[2,3])

    y4 = tf.cast(tf.square(tf.math.abs(y4)), dtype=tf.float64)
    
    y4 = tf.math.divide(tf.math.sqrt(y4), tf.math.sqrt(kappa))
    # print("Y4 - AT SHAPE", y4.shape, y4.dtype)
    # print("kappa - AT SHAPE", kappa.shape, kappa.dtype)
    return  y4

def AT_ASM_LAB(y, Masks, SFTF):
    # print("Y - AT SHAPE", y.shape, y.dtype)
    y = tf.cast(y, tf.complex128)
    #y = tf.multiply(tf.cast(y, tf.complex128), tf.math.exp(tf.zeros(y.shape, tf.complex128)))

    y1 = tf.signal.fftshift(y, axes=[2,3]);
    # print("Y1 - AT SHAPE", y1.shape, y1.dtype)
    y2 = tf.signal.fftshift(tf.signal.fft2d(y1), axes=[2,3]);
    # print("Y2 - AT SHAPE", y2.shape, y2.dtype)
    y3 = tf.signal.fftshift(tf.multiply(SFTF,y2), axes=[2,3]);
    # print("Y3 - AT SHAPE", y3.shape, y3.dtype)
    y_sftf = tf.signal.ifftshift(tf.signal.ifft2d(y3), axes=[2,3]);
    # print("Y_sftf - AT SHAPE", y_sftf.shape, y_sftf.dtype)
    res = tf.reduce_mean(tf.multiply(tf.math.conj(Masks), y_sftf), axis=1)
    # print("res - AT SHAPE", res.shape, res.dtype)
    #res = tf.multiply(Masks, y_sftf)
    return res
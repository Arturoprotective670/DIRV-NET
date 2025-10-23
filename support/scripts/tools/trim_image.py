from array import array
import numpy as np

def auto_trim_array(data : array,
                    intensity_threshold = 30,
                    count_threshold = 5,
                    tolerance_count = 3) -> array :
    """
    Automatically trims the a grayscale image array and removes the boarders of
    similar color.

    Arguments
    ---------
    - data: the monochrome image array.
    - intensity_threshold: intensity threshold, below which, all pixels are
      considered as background.
    - count_threshold: the maximum number of background pixels,
    that are allowed within one vertical/horizontal line, to allow its removal.
    - tolerance_count: the number of vertical/horizontal lines, that should be
    checked, before we decide that the border that can be removed is ended.

    Returns
    -------
    - An array of the trimmed image.
    """
    # below is a rough way to crop the images: we count the number of pixels
    # that has intensity higher than intensity_threshold (assuming that the
    # background is dark), per vertical and horizontal lines, starting from the
    # ages. if the number of such pixels, is less than count_threshold we
    # consider that this line is non-important, or a background, and we remove
    # it. if the current line happens to be not a background, we also make sure
    # that the next tolerance_count lines are also important, but if any of
    # those next lines is classified as background, we will remove the previous
    # important lines also. This is needed to avoid false positive important
    # lines that happens due to noise.

    # create a mask of pixels that has high intensity pixels
    mask_of_important_pixels = np.zeros(data.shape)
    mask_of_important_pixels[data > intensity_threshold] = 1

    # find the number of important pixels along rows
    important_pixels_counts = mask_of_important_pixels.sum(axis = 1)

    # verify background lines starting from the top edge
    tolerance_counter = 0
    for indx in range(len(important_pixels_counts)):
        if important_pixels_counts[indx] < count_threshold :
            data = data[tolerance_counter + 1 : , : ]
            tolerance_counter = 0
        else:
            tolerance_counter += 1
            if tolerance_counter >= tolerance_count:
                break

    # verify background lines starting from the bottom edge
    tolerance_counter = 0
    for indx in reversed(range(len(important_pixels_counts))):
        if important_pixels_counts[indx] < count_threshold :
            data = data[ : -(tolerance_counter + 1) , : ]
            tolerance_counter = 0
        else:
            tolerance_counter += 1
            if tolerance_counter >= tolerance_count:
                break

    # find the number of important pixels along columns
    important_pixels_counts = mask_of_important_pixels.sum(axis = 0)

    # verify background lines starting from the right edge
    tolerance_counter = 0
    for indx in range(len(important_pixels_counts)):
        if important_pixels_counts[indx] < count_threshold :
            data = data[ : , tolerance_counter + 1 : ]
            tolerance_counter = 0
        else:
            tolerance_counter += 1
            if tolerance_counter >= tolerance_count:
                break

    # verify background lines starting from the left edge
    tolerance_counter = 0
    for indx in reversed(range(len(important_pixels_counts))):
        if important_pixels_counts[indx] < count_threshold :
            data = data[ : , : -(tolerance_counter + 1)]
            tolerance_counter = 0
        else:
            tolerance_counter += 1
            if tolerance_counter >= tolerance_count:
                break

    if len(data) == 0 :
        print('WARNING: an empty image, after trimming, is detected!')

    return data

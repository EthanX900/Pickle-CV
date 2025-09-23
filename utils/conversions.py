
def convert_pixel_distance_to_feet(pixel_distance, reference_pixel_distance, reference_real_distance):
    if reference_pixel_distance == 0:
        return 0
    scale_factor = reference_real_distance / reference_pixel_distance
    return pixel_distance * scale_factor

def convert_feet_to_pixel_distance(feet_distance, reference_pixel_distance, reference_real_distance):
    if reference_real_distance == 0:
        return 0
    scale_factor = reference_pixel_distance / reference_real_distance
    return feet_distance * scale_factor
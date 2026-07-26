def get_room_shape_obstacles(shape_name, width, height):
    """
    Kthen një listë pengesash drejtkëndëshe që, kur kombinohen me
    dhomën bazë (width x height), krijojnë formën e kërkuar të dhomës.
    Pengesat funksionojnë si 'mure' shtesë brenda hapësirës drejtkëndëshe
    bazë, duke krijuar iluzionin e një forme jo-drejtkëndëshe.
    """
    if shape_name == "rectangle":
        return []

    elif shape_name == "l_shape":
        # Heq këndin e sipërm djathtas - krijon formë "L"
        return [(width * 0.5, 0, width * 0.5, height * 0.5)]

    elif shape_name == "t_shape":
        # Lë vetëm një "kollonë" qendrore + shiritin e sipërm - formë "T"
        return [
            (0, height * 0.35, width * 0.32, height * 0.65),
            (width * 0.68, height * 0.35, width * 0.32, height * 0.65),
        ]

    elif shape_name == "corridor":
        # Krijon një korridor të ngushtë horizontal në mes
        return [
            (0, 0, width, height * 0.38),
            (0, height * 0.62, width, height * 0.38),
        ]

    return []
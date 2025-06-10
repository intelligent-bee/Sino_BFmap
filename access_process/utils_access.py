
def speed_transform_road(x):
    if x=='secondary':
        return 1.5
    elif x=='primary':
        return 1
    elif x=='tertiary':
        return 6
    elif x=='trunk':
        return 1
    elif x=='motorway':
        return 0.5
    else:
        return 12
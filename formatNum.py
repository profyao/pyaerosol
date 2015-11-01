def orbit2str(orbit):
    return str(orbit).zfill(6)


def path2str(path):
    return str(path).zfill(3)


def get_key(date, path, orbit, block, p):

    date, path, orbit, block = get_date_key(date, path, orbit, block)
    return date, path, orbit, block, p


def get_date_key(date, path, orbit, block):
    return str(date), path2str(path), orbit2str(orbit), str(block)




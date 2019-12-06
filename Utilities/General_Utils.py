def get_on_path(obj:'Any', path:str, delimiter:str='.', getter:'Optional[Callable]'=None, **kwargs):
    """ 
    get in depth via str, 
    e.g. use 'a.b.c' to fetch as A.a.b.c
    """

    if getter is None:
        if isinstance(obj, dict):
            getter = type(obj).get
        else:
            getter = getattr

    for node in path.split(delimiter):
        obj = getter(obj, node, **kwargs)

    return obj
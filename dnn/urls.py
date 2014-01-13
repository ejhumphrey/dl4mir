
PARAM_MARKER = "."
NODE_MARKER = "/"


def _name_assert(inputs):
    for name in inputs:
        for marker in [NODE_MARKER, PARAM_MARKER]:
            assert not marker in name, \
                "'%s' may not contain '%s'!" % (name, marker)


def create(network, node, param):
    return "%s"*5 % (network, NODE_MARKER, node, PARAM_MARKER, param)


def parse(url):
    network, node, param = '', '', ''
    if NODE_MARKER in url:
        network, node_path = url.split(NODE_MARKER)
    else:
        node_path = url

    if PARAM_MARKER in node_path:
        node, param = node_path.split(PARAM_MARKER)
    else:
        param = node_path
    return network, node, param


def update(url, network="", node="", param=""):
    res = parse(url)
    for n, v in enumerate([network, node, param]):
        if v:
            res[n] = v
    return create(res[0], res[1], res[2])


def append_param(base, param):
    assert not PARAM_MARKER in base
    return "%s%s%s" % (base, PARAM_MARKER, param)


def split_param(url):
    if PARAM_MARKER in url:
        return url.split(PARAM_MARKER)
    return url, ""


def append_node(base, node):
    assert not NODE_MARKER in base
    return "%s%s%s" % (base, NODE_MARKER, node)

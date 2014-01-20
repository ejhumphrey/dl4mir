
PARAM_MARKER = "."
INPUT_MARKER = "$:"
OUTPUT_MARKER = "=:"


def create(node, param):
    return "%s"*3 % (node, PARAM_MARKER, param)


def parse(url):
    node, param = '', ''
    if PARAM_MARKER in url:
        node, param = url.split(PARAM_MARKER)
    else:
        param = url
    return node, param


def update(url, node="", param=""):
    res = parse(url)
    for n, v in enumerate([node, param]):
        if v:
            res[n] = v
    return create(res[0], res[1])


def append_param(base, param):
    assert not PARAM_MARKER in param
    return "%s%s%s" % (base, PARAM_MARKER, param)


def split_param(url):
    if PARAM_MARKER in url:
        return url.split(PARAM_MARKER)
    return url, ""


def node(url):
    return parse(url)[1]


def param(url):
    return parse(url)[2]


def is_input(url):
    return url.startswith(INPUT_MARKER)


def parse_input(url):
    assert is_input(url)
    return url.strip(INPUT_MARKER)


def is_output(url):
    return url.startswith(OUTPUT_MARKER)


def parse_output(url):
    assert is_output(url)
    return url.strip(OUTPUT_MARKER)

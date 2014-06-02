import argparse
import urllib2
import os.path as path
import time
import json


DOMAIN = "http://www.e-chords.com/%s"


def fetch(relative_url):
    request_result = urllib2.urlopen(DOMAIN % relative_url)
    return request_result.read()


def main(args):
    queries = json.load(open(args.query_file))
    for key, metadata in queries.iteritems():
        if len(queries[key]['chord_urls']) == 0:
            continue
        min_len = 1000000
        url = ''
        for a_url in queries[key]['chord_urls']:
            if len(a_url) < min_len:
                url = a_url
                min_len = len(a_url)
        artist, title = metadata['artist'], metadata['title']
        output_file = path.join(args.output_directory, "%s.html" % key)
        if path.exists(output_file) and args.skip_existing:
            continue
        print "[%s] Fetching: %s - %s" % (time.asctime(), artist, title)
        page_contents = fetch(url)

        with open(output_file, 'w') as fp:
            fp.write(page_contents)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download a collection of webpages.")
    parser.add_argument(
        "query_file",
        metavar="query_file", type=str,
        help="A JSON file of objects with 'chord_urls' values.")
    parser.add_argument(
        "--skip_existing",
        metavar="skip_existing", type=bool,
        default=False,
        help="Length of the median filter (must be odd).")
    parser.add_argument(
        "output_directory",
        metavar="output_directory", type=str,
        help="Output path for downloaded webpages.")
    main(parser.parse_args())

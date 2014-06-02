import argparse
import requests
from bs4 import BeautifulSoup
import urllib
import json
import time

QUERY_FMT = "http://www.e-chords.com/search-all/%s/show-chords"
NO_RESULTS = "0 results"


def form_url_query(artist, title):
    query_text = u'%s %s' % (artist, title)
    return QUERY_FMT % urllib.quote(query_text.encode('utf8'))


def get_results_urls(soup):
    results = set()
    for link in soup.findAll("a"):
        if link.attrs.get("href", "").startswith("chords/"):
            results.add(link.attrs.get("href"))

    return list(results)


def find(artist, title):
    query = form_url_query(artist, title)
    request_result = requests.get(query)
    if NO_RESULTS in request_result.text:
        return list()
    soup = BeautifulSoup(request_result.text)
    search_results = soup.find("ul", attrs={"id": "results"})
    return get_results_urls(search_results)


def main(args):
    queries = json.load(open(args.query_file))
    for key, metadata in queries.iteritems():
        if queries[key].get('chord_urls', None):
            continue
        try:
            artist, title = metadata['artist'], metadata['title']
            print "[%s] Query: %s - %s" % (time.asctime(), artist, title)
            queries[key]['chord_urls'] = find(artist, title)
            print "\tFound %d result(s)" % len(queries[key]['chord_urls'])
        except:
            print "\toh noes, failed :o("
    with open(args.output_file, 'w') as fp:
        json.dump(queries, fp, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Make a set of queries and write the results as JSON.")
    parser.add_argument(
        "query_file",
        metavar="query_file", type=str,
        help="A JSON file of objects with 'artist' and 'title' values.")
    parser.add_argument(
        "output_file",
        metavar="output_file", type=str,
        help="File for writing search results.")
    main(parser.parse_args())

"""
Assemble a list of URLs to Flickr images fetched by repeated calls to
the API method flickr.interestingness.getList.

To use, you must place your [API key](http://www.flickr.com/services/apps/72157632345167838/key/)
into a file named 'flickr_api.key' located in the same directory as this file.

There is a limit of 500 images per day, so to obtain more images than that,
we iterate backwards from the current date until the desired number of images
is obtained.
"""

import datetime
from urllib2 import urlopen
import simplejson as json
import gzip
import operator
import re


def get_url(photo):
    # see http://www.flickr.com/services/api/misc.urls.html for size options
    size = 'm'  # 240px on the longest side
    url = "http://farm{farm}.staticflickr.com/{server}/{id}_{secret}_{size}.jpg"
    return url.format(size=size, **photo)


def get_photos_list(api_key, date):
    params = {'api_key': api_key, 'date': str(date)}
    params['date'] = str(date)
    params['per_page'] = 500  # 500 maximum
    url = "http://api.flickr.com/services/rest/?method=flickr.interestingness.getList&api_key={api_key}&date={date}&per_page={per_page}&format=json&nojsoncallback=1"
    url = url.format(**params)
    print(url)
    data = json.load(urlopen(url))
    if not data['stat'] == 'ok':
        raise Exception("Something went wrong: API returned status {}".format(data['stat']))
    return data


def assemble_flickr_dataset(api_filename, data_filename, num_images_to_load):
    """
    Assemble dataset containing the specified number of images using Flickr
    'interestingness' API calls.
    Returns nothing; writes data to file.

    Parameters
    ----------
    api_filename : string
        File should contain only one line, with your Flickr API key.
    data_filename : string
        Gzipped JSON file that will contain the dataset.
        If it already exists, we will load the data in it and not repeat
        the work done.
    num_images_to_load : int
    """
    with open(api_filename) as f:
        api_key = f.readline().strip()
    
    try:
        with gzip.open(data_filename) as f:
            urls_by_date = json.load(f)
    except:
        urls_by_date = {}
    
    # Begin from yesterday and go backwards
    one_day = datetime.timedelta(days=1)
    date = datetime.date.today() - one_day

    print("Loading {} images".format(num_images_to_load))
    num_loaded = sum((len(urls) for date, urls in urls_by_date.iteritems()))
    while num_loaded < num_images_to_load:
        if str(date) in urls_by_date:
            date -= one_day
            continue
        
        data = get_photos_list(api_key, date)
        urls = [get_url(photo) for photo in data['photos']['photo']]
        urls_by_date[str(date)] = urls
        
        num_loaded = sum((len(urls) for date, urls in urls_by_date.iteritems()))
        print("Finished {0} with {1} images loaded".format(date, num_loaded))
        date -= one_day
    with gzip.open(data_filename, 'wb') as f:
        json.dump(urls_by_date, f)


def ids_and_urls_from_dataset(data_filename, num_images):
    """
    Load the data in given filename and return the first num_images urls
    (or all of them if num_images exceeds the total number).

    Parameters
    ----------
    data_filename : string
        JSON file that will contain the dataset.
    num_images : int

    Returns
    -------
    ids : list
        the Flickr IDs of the photos
    urls : list
        URLs of the photos
    """
    with gzip.open(data_filename) as f:
        urls_by_date = json.load(f)
    urls = reduce(operator.add, [urls for urls in urls_by_date.values()])

    def get_id(url):
        _id = re.search('flickr.com/\d+/(.+)_.+_.+.jpg', url).groups()[0]
        return 'flickr_{}'.format(_id)
    ids = [get_id(url) for url in urls]
    assert(len(set(ids)) == len(ids))
    return ids[:num_images], urls[:num_images]


if __name__ == '__main__':
    """
    The API key file and the data output files are expected to be
    in the currently active directory.
    """
    api_filename = 'flickr_api.key'
    data_filename = 'data/flickr_1M.json.gz'
    num_images_to_load = int(1e6)
    assemble_flickr_dataset(api_filename, data_filename, num_images_to_load)

Quick start
-----------

To get your Rayleigh running locally, you must first populate a SearchableImageCollection.

You can download a sample .pickle file and try running the server: ::

    wget -P data https://s3.amazonaws.com/rayleigh/data/flickr_100K_exact_chi_square_16_0.pickle

To run Rayleigh, we need to have a mongodb server running: ::
    
    mongod --dbpath db --port 27666

In another shell tab, run ::

    python rayleigh/client/app.py

You should now be able to access the running website at http://127.0.0.1:5000/

To construct your own dataset from scratch, run ::

    nosetests test/collection.py:TestFlickrCollection -s

This uses the file data/flickr_1M.json.gz, which lists a million images from Flickr 
fetched by the "interestingness" API query over the last few years.

Running this will download and process 100K images (or less, or more, if you modify the code).
Data is stored into the mongodb database.
It will help to have multiple cores working, so in a separate tab, do ::

    ipcluster start 8

This relies on the IPython parallel framework.

If you want, you can reoutput the Flickr URLs ::

    python rayleigh/assemble_flickr_dataset.py

Have fun!
This is a part of Academic Project under the course Winter Training (COE-319) at NSIT.
## Fun Lyrics
A  recurrent neural network (RNN) based project to generate lyrics for an artist.

## Usage
Change the artist name and url in config.py

```python
# change artist name and url
ARTIST_NAME = ""
URL = ""
```
Then run these commands:
```sh
python getLyrics.py
python model.py
python generate.py
```
## Example

```
Uh yo, yo
Doesn't matter who you are or where you're from
So live do wittoh
Make she fly with the strukes it on back
Just livin' my life  my life
Now there's gravel in our voices
Make she fly with the strukes it on back
They don't messent, can't seem to find a lot
Doesn't matter who you are or where you're from
So live do wittoh
For all the clos
Hey, hey, hey hey, hey, hey,
Let me gon' wan to trust in like a say the way
Every time my love you
Cause I don't wanna be alone
And save me from myself and all this conflict
Doesn't matter who you are or where you're from
Doesn't matter who you are or where you're from
I wish you would push up on me
Hey, hey, hey hey, hey, hey,
Every time my love you
So live do wittoh
Cause I don't wanna be alone

```
This can be used to train any other text on LSTM  network and the generate/predict text.
The configuration of the network can be changed in config.py.
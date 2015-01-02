import numpy as np
import string
import time

import biggie

import dl4mir.chords.decode as D
import dl4mir.chords.lexicon as lex

NUM_CPUS = 12


def create_entity(num_samples=3000, num_states=157):
    posterior = np.random.normal(size=(num_samples, num_states))

    return biggie.Entity(
        posterior=posterior - posterior.min(),
        time_points=np.arange(num_samples) / 20.0)


def main():
    vocab = lex.Strict(157)
    stash = {k: create_entity() for k in string.ascii_letters}

    print "[{0}] Testing decode_posterior".format(time.asctime())
    D.decode_posterior(stash['a'], -10.0, vocab)

    print "[{0}] Testing decode_posterior_parallel".format(time.asctime())
    penalties = np.linspace(-1, -40, 25)
    D.decode_posterior_parallel(stash['a'], penalties, vocab, NUM_CPUS)

    print "[{0}] Testing decode_stash_parallel".format(time.asctime())
    D.decode_stash_parallel2(stash, -10.0, vocab, NUM_CPUS)

    print "[{0}] Done!".format(time.asctime())

if __name__ == "__main__":
    main()

"""The laning brain: what RiftWatch learns from high-elo gameplay videos.

The video library turns each replay into readings four times a second through the laning
phase (both health bars, spacing, minions, mana, abilities ready, level, map position) and a
list of trades. The brain learns from those readings, in five parts:

* ``features`` -- the catalogue of what the brain sees at a moment: the readings themselves,
  plus their recent history (who has been losing health, how long since the last trade, how
  long ago a level-up landed, whether the two are closing in). Each feature names the group it
  belongs to (health, wave, abilities, ...) for importance and explanations, and whether only
  replays show it (the HUD panel and minimap aren't read from your own recordings).
* ``data`` -- every moment a player stands within trading range of their lane opponent,
  outside a trade, is a *situation*; ``data`` builds them from the database with their
  features and the labels for every head, and checks each video's readings before it is
  trusted for training.
* ``heads`` -- the questions the brain answers about a situation: does a high-elo player start
  a trade here (the policy), does the opponent start one on them (the threat), and for trades
  started from here how they go (won or not, net health) and what follows (the health swing
  over the next ten seconds, dying, being back in base).
* ``train`` -- every head is fitted and judged on whole held-out games (grouped
  cross-validation), against the plain base rate it has to beat; a head that doesn't beat it
  consistently is kept for the record but marked unusable, and the coach never quotes it.
  Training also picks hyperparameters, calibrates the probabilities, bags the final model
  over resampled games (so its uncertainty is known), measures which feature groups matter,
  draws a learning curve (is more video still helping?) and extracts the patterns it learned
  in plain words. Role-specific models replace the pooled one where they do better.
* ``review`` -- the brain applied to one video (usually your own game): over every spot in
  range of your opponent, how many trades high-elo players would have started against how
  many you did, and the key moments -- trades high-elo players would have taken that you
  didn't, trades you took that they rarely do and that went badly, spots where they expected
  you to get hit -- each with the reasons that drove the brain's view.

``registry`` keeps every trained brain with its model card; the newest becomes current
unless told otherwise, and any earlier one can be switched back to.

The heads learn associations in high-elo play, not causes: players trade when a trade is
already likely to go well. Evidence is phrased that way ("in spots like this high-elo players
..."), and outcome numbers are only quoted from heads that earned it.
"""

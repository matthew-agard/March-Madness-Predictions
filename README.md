# Creating the Perfect Bracket

There's nothing quite like the most riveting basketball event of the year: NCAA March Madness. The 64-team tournament consists of 4 regions, each with 16 teams ranked independently of the other regions according to their regular season performance. Each team attempts to win 6 successive games in order to emerge victorious as the NCAA national champion.

Perhaps what contributes most to the intrigue of March Madness is filling out a March Madness bracket. "The American Gaming Association estimated in 2019 that 40 million Americans filled out a combined 149 million brackets for a collective wager of \$4.6 billion." It's important to note that even a single bet can be quite lucrative, particularly when an upset occurs (when a lower-ranking underdog beats a higher-ranking favorite). For example, the first-ever upset of a #1 seed by a #16 seed occurred in the 2019 NCAA tournament. In that game "a \$100 bet paid out \$2,500", which translates to American betting odds of +2500!

All quotations were cited from the following article:
https://www.gobankingrates.com/money/business/money-behind-march-madness-ncaa-basketball-tournament/

# Problem Structure

The purpose of this personal project is to perform supervised classification on March Madness data to more accurately predict the outcome of an NCAA tournament games--particularly the occurrence of upsets. This would allow for an increased possibility of yielding the kinds of profits mentioned above by filling out more accurate brackets relative to other participants.

# TODO
- Add EDA question for conferences' historical performance in the tournament
- Feature Tasks:
    - Rewrite data_fetch code to fetch current bracket from sportsreference (potentially 2023-beta)
    - Explore potential use of removed null features in model predictions
- Models:
    - Neural Network
- Metrics:
    - Log Loss
    - Hinge Loss
    - Etc... (research more ideas)
- Productionize model to AWS cloud; brainstorm services to use
    - S3
    - Lambda
    - Glue
    - Athena/Redshift
    - SageMaker

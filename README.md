# CIS419-Final-Project-Text-Classification



This the Final Project for CIS419. 

By Neil Shweky, Hersh Solanki, and Kamal El Bayrouti.

Our Project Proposal:


### What are you trying to do? Articulate your objectives using absolutely no jargon.    
We are trying to predict the accuracy of people’s reactions to Earnings Calls. An Earnings Call is a phone call in which a company reports on their financial earnings for the quarter. We can do this by analyzing past earnings calls and their results, and then predicting how we think other calls will reflect on a company’s stock. We will then determine if the market over or underreacted, and advise whether or not to invest.


### How is it done today, and what are the limits of current practice?
This is a strategy that a lot of quantitative funds run as a signal in systematic trading; their methods are not made publicly available. Based on publicly available research, text regression and "bag of words" methods are the most popular in understanding the text itself. The biggest limitation to using ML in the markets is that the markets are inherently random, with order of random inputs leading to a random output. Thus, while it hard to predict prices, it is more achievable to predict a binary overreaction.


### What's new in your approach and why do you think it will be successful?
The novelty in our approach lies in the use of NLP on transcripts to tease out details that cause market overreaction. We reason that overreaction is largely driven by people's sentiments, which are largely driven by what people hear and read. We think that given this, and the narrow scope of prediction, the learner maybe able to detect meaningful patterns.


### Who cares? If you're successful, what difference will it make?
If we are successful, we have a legitimate trading strategy that we/you can implement in order to make consistent returns over the long run. Even a 1 \% increase in hit rate, or the amount of correct predictions out of 100, can lead to exponential returns when taking advantage of leverage.


### What are the risks and the payoffs? 
The greatest risk of the model is also the its greatest payoff: investor returns. The investor stands to profit if the market behaves in line with the model’s expectations, otherwise, the investor loses money. Furthermore, if an investor chooses not to enter a position following the model’s suggestion, the investor either avoids losses, or misses out on profits (opportunity cost).


### What are the midterm and final "exams" to check for success?
The midterm “exam” would be to check that we have all of the data cleaned, and that we have all of the necessary features to make our prediction. The final “exam” would be to actually make predictions on earning calls and see the accuracy. 
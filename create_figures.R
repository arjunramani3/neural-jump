#Create Figures
install.packages('tidyverse')
install.packages('ggplot2')
library(ggplot2)
library(tidyverse)
df <- read_csv('/Users/arjun/Documents/cs224n/deepjump/jumps_by_day.csv')

sub <- df[, 10:26]

sub['max'] <- colnames(sub)[max.col(sub,ties.method="first")]
dist <- as.data.frame(table(sub$max))
dist <- dist[order(-dist$Freq), ]

# Basic histogram
ggplot(dist, aes(x=Var1, y=Freq)) + geom_bar(stat='identity')
# Change the width of bins

# Change colors
a<-ggplot(dist, aes(x=Var1, y=Freq)) + 
  geom_bar(stat='identity', color="black", fill="white") + 
  labs(x = 'Category', y = 'Number of Jumps') + 
  theme(text = element_text(size=15),
        axis.text.x = element_text(angle = 45))
a


dist2 <- dist[1:6,]
p<-ggplot(dist2, aes(x=Var1, y=Freq)) + 
  geom_bar(stat='identity', color="black", fill="white") + 
  labs(x = 'Category', y = 'Number of Jumps') + 
  theme(text = element_text(size=18),
        axis.text.x = element_text(angle = 45))
p

dist3 <- dist2
dist3$Freq = dist3$Freq*2
dist3$Freq[1] = dist2$Freq[dist2$Var1 == 'Macro']

q<-ggplot(dist3, aes(x=Var1, y=Freq)) + 
  geom_bar(stat='identity', color="black", fill="white") + 
  labs(x = 'Category', y = 'Number of Jumps') + 
  theme(text = element_text(size=18),
        axis.text.x = element_text(angle = 45))
q

dist4 = dist2[-2, ]
dist4$frac = dist4$Freq/sum(dist4$Freq)
dist4$frac2 <- dist4$frac^2
sum(dist4$frac2)



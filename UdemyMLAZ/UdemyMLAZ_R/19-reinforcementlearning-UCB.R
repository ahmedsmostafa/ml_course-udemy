dataset = read.csv("..\\ReinforcementLearning_UpperConfidenceBound\\Ads_CTR_Optimisation.csv", header = FALSE)

#implement UCB
N = nrow(dataset) - 1 #remove the header row from the count
d = length(dataset) #how many ads
ads_selected = integer(0)
count_of_selection = integer(d)
sums_of_rewards = 0
reward = 0
total_reward = 0
for (n in 1:N) {
    max_upper_bound = 0
    ad = 0
    for (i in 1:d) {
        upper_bound = 0
        if (count_of_selection[i] > 0) {
            average_reward = sums_of_rewards[i] / count_of_selection[i]
            delta_i = sqrt(3 / 2 * log(n) / count_of_selection[i])
            upper_bound = average_reward + delta_i
        } else {
            upper_bound =  1000000
        }
        
        if (upper_bound > max_upper_bound) {
            max_upper_bound = upper_bound
            ad = i
        }
    }
    ads_selected = append(ads_selected, ad)
    count_of_selection[ad] = count_of_selection[ad] + 1
    reward = dataset[n, ad]
    sums_of_rewards[ad] = sums_of_rewards[ad] + reward
    total_reward = total_reward + reward
}

hist(ads_selected,
     col = 'blue',
     main = 'Histogram of ads selections',
     xlab = 'Ads',
     ylab = 'Number of times each ad was selected')


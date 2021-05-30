# Risks in using Intermediate Data for Domain Transfer Between SimilarDomains

As  unlabelled  data  sets  become  more  com-mon, the methods for learning from them be-come  more  relevant.   Semi-supervised  learn-ing, and more specifically self-training, is onesuch type of method applicable when labelleddata from a similar domain is available.  If thedomain differences are sufficiently large, inter-mediate  data,  similar  to  both  source  and  tar-get, might sustain the generalisation across thetransfer,  according  to  new  research.   We  ex-plore whether there are risks to this approach,by  using  it  on  domains  that  have  small  dif-ferences and thus cannot be expected to ben-efit from the procedure.  We observe that un-der  certain  circumstances,  our  model  is  in-deed  negatively  affected  by  the  inclusion  ofintermediate data.  These findings are not par-ticularly surprising,  as introducing intermedi-ate  data  between  domains  that  already  haveconsiderable overlap, seems likely to only be-come  a  source  of  noise.  

library(ggplot2)
library(pROC)
##AUC####
roc <- roc(df$Type, df$value  ,show.thres = T, plot = T,print.thres="best", thresholds="best")

#####box plot###
colors <- c("#1B8F8E","#0000FF", "#FF0000" , "#BCBD22", "#D37132", "#800080", "#E377C2")#,"#8C564B", "#17BECF"

parameter <- c(0.0,0.8,0.1) 
jpeg(quality = 100, res = 600,file = path) 
p <- ggplot(df_bar1, aes_string(x = "Group_ID", y = "CRI", group = "Group_ID", color = "Group_ID")) +
  geom_jitter(size = 1*word_cex, alpha=0.7, width = 0.15*length_cex, height = 0) +
  stat_boxplot(geom = "errorbar", width = 0.2*length_cex, size = 1*length_cex) + 
  geom_boxplot(width = 0.4*length_cex, size = 1*length_cex,outlier.colour = NA,alpha = 0.0) +
  
  
  scale_color_manual(values=colors) + 
  
  theme_bw()  +
  scale_y_continuous(limits=c(parameter[1], parameter[2]), expand = expansion(mult = c(0.05, 0)),breaks = seq(parameter[1],parameter[2],parameter[3])) +
  labs(x=NULL, y = "Cancer risk index") +  
  theme( 
    panel.grid = element_blank(),            
    panel.border = element_blank(),
    legend.position = "none",
    legend.text = element_text(size = 10*word_cex*n),  
    legend.title = element_text(size  = 0) ,  
    axis.title.x = element_text(size = 10*word_cex*n),   
    axis.title.y = element_text(size = ylab_size*word_cex*n, vjust = 0.5, margin = margin(0,0.5,0,0,'cm')),  
    axis.text.x  = element_text(size = 8*word_cex*n, hjust = 0.5,color = "black", angle = 0), 
    axis.text.y  = element_text(size = 8*word_cex*n,color = "black"),  
    axis.line.x = element_line(size = 1*length_cex*n), 
    axis.line.y = element_line(size = 1*length_cex*n), 
    axis.ticks.length = unit(1*n, "pt"),     
    axis.ticks = element_line(color="black",size = 1*length_cex*n),     
    plot.margin = unit(c(2*n,0,0,0), "pt")
  )
print(p)
dev.off() 






















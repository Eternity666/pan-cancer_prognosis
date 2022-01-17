import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

# myfont = FontProperties(fname=r'C:\Windows\Fonts\simhei.ttf')
# sns.set(font=myfont.get_family())
# sns.set_style("whitegrid", {"font.sans-serif": ['Source Han Sans CN']})
sns.set_style('whitegrid')
plt.rcParams['font.sans-serif'] = ['simhei']
plt.rcParams['axes.unicode_minus'] = False
sns.set(font=['simhei', 'Arial'], font_scale=1.5)

df = pd.read_excel(r"F:\res\identity.xlsx")
ax = sns.boxplot(x='数据模态', y='一致性指数', data=df, hue='残差学习', width=0.4,
                 palette=sns.color_palette("husl"))

# # Select which box you want to change
# mybox = ax.artists[2]
#
# # Change the appearance of that box
# mybox.set_facecolor('pink')
# mybox.set_edgecolor('green')
# mybox.set_linewidth(3)

plt.show()

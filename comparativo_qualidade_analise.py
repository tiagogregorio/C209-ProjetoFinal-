import pandas as pd
import matplotlib.pyplot as plt

avaliacao = pd.read_csv('C:/Users/Eniac/Downloads/Curso_Subtraction-20231012T224441Z-001/Curso_Subtraction/videos/report.csv')
#print(avaliacao)

resultado_avaliacao = avaliacao.groupby(['Frame']).sum()
print(resultado_avaliacao)

resultado_avaliacao.plot.bar()
plt.show()

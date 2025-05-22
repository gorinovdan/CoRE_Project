import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Загрузка и подготовка данных
data = pd.read_csv("CoRE_Dataset/CoRE_Dataset/unified_dataset.csv")
df = pd.DataFrame(data)

# Список правил композиции
composition_rules = df.columns.tolist()[4:]

# Подсчёт количества изображений, у которых правило включено (1.0)
counts = (df[composition_rules] > 0).sum().sort_values(ascending=False)

# Компактный красивый вывод в консоль
print("\nРаспределение изображений по правилам композиции:\n")
for rule, count in counts.items():
    print(f"• {rule:<25} — {count:>5}")

# Построение графика
plt.figure(figsize=(12, 6))
sns.set(style="whitegrid")
palette = sns.color_palette("crest", len(counts))

plot_df = pd.DataFrame({
    "Rule": counts.index,
    "Count": counts.values
})

ax = sns.barplot(data=plot_df, x="Rule", y="Count", hue="Rule", palette=palette)

# Удаление легенды
legend = ax.get_legend()
if legend:
    legend.remove()

plt.title("Количество изображений с каждым правилом композиции", fontsize=16, weight='bold')
plt.ylabel("Количество изображений", fontsize=12)
plt.xlabel("Правило композиции", fontsize=12)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

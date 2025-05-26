import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('students.csv')
print(df)

df['focused_rate'] = df['focused_time']/df['total_time'] * 100

courses = df.groupby('course')
lecturers = df.groupby('lecturer')
classes = df.groupby('class')


courses_focus = df.groupby('course')['focused_rate'].mean()
print(courses_focus)
lecturer_focus = df.groupby('lecturer')['focused_rate'].mean()
print(lecturer_focus)
classes_focus = df.groupby('class')['focused_rate'].mean()
print(classes_focus)


sns.set_style("whitegrid")
sns.set_theme(style="ticks")
sns.set_palette("Spectral")
plt.figure(figsize=(13, 5))

ax1 = plt.subplot(1, 3, 1)
sns.barplot(x=courses_focus.index, y=courses_focus.values, palette="RdPu", ax=ax1)
sns.despine()
plt.title('Average Focus Rate by Course')
plt.xlabel('Course')
plt.ylabel('Average Focused Rate (%)')
plt.xticks(rotation=90, ha='right')
plt.ylim(0, 100)
for p in ax1.patches:
    ax1.annotate(f'{p.get_height():.1f}%', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points', fontsize=9)

ax2 = plt.subplot(1, 3, 2)
sns.barplot(x=lecturer_focus.index, y=lecturer_focus.values, palette="Reds", ax=ax2)
sns.despine()
plt.title('Average Focus Rate by Lecturer')
plt.xlabel('Lecturer')
plt.ylabel('Average Focused Rate (%)')
plt.xticks(rotation=90, ha='right')
plt.ylim(0, 100)
for p in ax2.patches:
    ax2.annotate(f'{p.get_height():.1f}%', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points', fontsize=9)

ax3 = plt.subplot(1, 3, 3)
sns.barplot(x=classes_focus.index, y=classes_focus.values, palette="Wistia", ax=ax3)
sns.despine()
plt.title('Average Focus Rate by Class')
plt.xlabel('Class')
plt.ylabel('Average Focused Rate (%)')
plt.xticks(rotation=90, ha='right')
plt.ylim(0, 100)
for p in ax3.patches:
    ax3.annotate(f'{p.get_height():.1f}%', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 10), textcoords='offset points', fontsize=9)

plt.tight_layout()
plt.show()
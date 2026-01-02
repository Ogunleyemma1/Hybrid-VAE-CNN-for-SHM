plt.figure(figsize=(12, 8))
plt.style.use('fivethirtyeight') # you can keep optional
plt.plot(range(bo_iterMax + 1), median, label=method['name'], color=method['color'], linewidth=6)
plt.xlabel("Iteration", fontsize=20)  # 25 as default slide fonts
plt.ylabel(r"$\log(y_{\min} - f^*)$", fontsize=20)
plt.xticks(ticks=range(0, 1001, 200), fontsize=20)
plt.yticks(fontsize=20)
plt.xlim(0, bo_iterMax)
plt.legend(fontsize=15, loc='upper right')
plt.tight_layout()
save_path_pdf = f"{function_name.replace(' ', '_')}_Optimization_Histories_Median_IQR_Slide.pdf"
save_path_svg = save_path_pdf.replace(".pdf", ".svg")
 
plt.savefig(save_path_pdf, format='pdf', bbox_inches='tight')
plt.savefig(save_path_svg, format='svg', bbox_inches='tight')
plt.show()
print(f"Median + IQR plot saved as both '{save_path_pdf}' and '{save_path_svg}'.")
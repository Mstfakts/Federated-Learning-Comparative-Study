from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

FILE_NAME = "TumDeneySonuclari.xlsx"
here = Path(__file__).resolve()
project_root = here.parents[2]
results_dir = project_root / "results" / "ml_pipeline_experiments" / "experimental_results" / FILE_NAME


def load_all_sheets(file_path):
    xls = pd.ExcelFile(file_path)
    df_list = []
    for sheet in xls.sheet_names:
        df_sheet = pd.read_excel(xls, sheet_name=sheet)
        df_list.append(df_sheet)

    df_res = pd.concat(df_list, ignore_index=True)
    return df_res


df = load_all_sheets(results_dir)
df["Dataset"] = df["Dataset"].replace("TAIWAN", "DCCC")
df["Feature Reduction"] = df["Feature Reduction"].replace("N", "None")
df["Data Sampling"] = df["Data Sampling"].replace("N", "None")


def create_final_static_plot(metric_name):
    """
    Tüm senaryoları içeren, başlıkları temizlenmiş, statik bir 3x3 grid üzerinde,
    veri örneklemenin etkisini bağlantılı nokta grafiği ile gösterir.
    """
    df_filtered = df[df['Feature Reduction'] == 'None'].copy()

    # Sıralamayı manuel olarak tanımlayarak tutarlılığı sağlıyoruz
    aggregations = ['Centralized', 'FedAvg', 'FedF1']
    datasets = ['DCCC', 'HCDR', 'HMEQ']
    ml_models = ['LR', 'MLP', 'SVM', 'XGB', 'RF']
    colors = {"None": "skyblue", "SMOTE": "mediumseagreen", "RUS": "salmon"}

    # 3x3'lük bir matplotlib figürü ve alt grafik (axes) dizisi oluşturalım
    fig, axes = plt.subplots(
        nrows=len(aggregations),
        ncols=len(datasets),
        figsize=(18, 11),
        sharey=True,
        sharex=True
    )

    # Genel başlığı kaldırıp, daha hassas konumlandırma için fig.text kullanacağız
    # fig.suptitle(f"Veri Örnekleme Tekniklerinin {metric_name} Skoruna Etkisi", fontsize=24, y=0.98)

    # Her bir alt grafiği dolduralım
    for i, agg in enumerate(aggregations):
        for j, ds in enumerate(datasets):
            ax = axes[i, j]
            plot_data = df_filtered[(df_filtered['Aggregation'] == agg) & (df_filtered['Dataset'] == ds)]

            if plot_data.empty:
                ax.text(0.5, 0.5, 'Veri Yok', ha='center', va='center')
                continue

            # Dambıl çizgileri
            for model_name in ml_models:
                model_data = plot_data[plot_data['ML Model'] == model_name]
                vals = {s_type: model_data[model_data['Data Sampling'] == s_type][metric_name].values for s_type in
                        ["None", "SMOTE", "RUS"]}

                valid_vals = [v[0] for v in vals.values() if len(v) > 0]
                if len(valid_vals) > 1:
                    ax.hlines(y=model_name, xmin=min(valid_vals), xmax=max(valid_vals), color='lightgrey', lw=1.5,
                              zorder=1)

            # Noktalar
            sns.stripplot(
                data=plot_data, x=metric_name, y='ML Model', hue='Data Sampling',
                order=ml_models, hue_order=["None", "SMOTE", "RUS"], palette=colors,
                size=8, edgecolor='gray', linewidth=0.5, ax=ax, zorder=2
            )

            ax.grid(axis='x', linestyle='--', alpha=0.7)
            ax.set_title("")

            # Sadece en soldaki grafiklere SATIR başlığı (Aggregation) ekle
            if j == 0:
                ax.set_ylabel(agg, fontsize=16, labelpad=25, rotation=0, ha='right', va='center')
            else:
                ax.set_ylabel("")

            # X ekseni etiketlerini (Dataset) sadece en alttaki satır için göster
            if i == len(aggregations) - 1:
                ax.set_xlabel(ds, fontsize=16, labelpad=15)
            else:
                ax.set_xlabel("")
                ax.tick_params(axis='x', labelbottom=False)

    # Ortak X ekseni limiti
    plt.xlim(-0.05, 1.05)

    # --- YENİ DÜZENLEME: BAŞLIK VE LEGEND'İ ORTADAKİ SÜTUNA HİZALAMA ---

    # 1. Adım: Legend bilgilerini al ve tüm bireysel legendları temizle
    handles, labels = axes[0, 0].get_legend_handles_labels()
    for ax_row in axes:
        for ax_col in ax_row:
            if ax_col.get_legend() is not None:
                ax_col.get_legend().remove()

    # 2. Adım: Başlığı ortadaki sütunun üstüne yerleştir
    # Ortadaki sütunun merkezini x koordinatı olarak alıyoruz (yaklaşık 0.55)
    fig.text(0.55, 0.98, f"Veri Örnekleme Tekniklerinin {metric_name} Skoruna Etkisi",
             ha='center', va='center', fontsize=24)

    # 3. Adım: Legend'ı başlığın altına, yine ortadaki sütuna hizalı şekilde yerleştir
    fig.legend(handles, labels, title="Örnekleme Tekniği",
               loc="upper center",
               bbox_to_anchor=(0.55, 0.92),  # x=0.55 ile ortala
               ncol=3, fontsize=12, title_fontsize=13)

    # Figürün genel yerleşimini hassas bir şekilde ayarlayalım
    # DÜZELTME: Sol boşluk ayarlandı
    fig.subplots_adjust(left=0.17, right=0.98, top=0.85, bottom=0.15, hspace=0.1, wspace=0.05)

    plt.show()


# --- ANA ÇALIŞTIRMA KISMI ---
# Sadece en önemli hikaye olan Recall için grafiği çizdirelim.
metrics_to_plot = ['Recall']

for metric in metrics_to_plot:
    create_final_static_plot(metric)
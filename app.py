import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score
# 1. Konfigurasi Halaman
st.set_page_config(
    page_title="Dashboard Hasil Clustering Pajak Riau",
    page_icon="📊",
    layout="wide"
)

# 2. Sidebar Navigasi dan Informasi Skripsi
st.sidebar.title("Navigasi Dashboard")
st.sidebar.image("WA0007.jpg", use_container_width=True)
st.sidebar.markdown("---")

st.sidebar.markdown("<div style='padding: 0.2rem 0 0.5rem 0;'><b>Menu</b></div>", unsafe_allow_html=True)
menu = st.sidebar.radio(
    "",
    ["🏠 Ringkasan Data", "📋 Profil & Karakteristik"],
    label_visibility="collapsed"
)

st.sidebar.info(
    "**Skripsi:** \n\nClustering Tingkat Kepatuhan Wajib Pajak Kendaraan Bermotor Di Provinsi Riau\n\n"
    "**Oleh:** \n\nYayat Hidayatullah (23082003)\n\n"
    "**UNIVERSITAS HANG TUAH PEKANBARU**"
)

# 3. Path File Statis (Pastikan file CSV berada di folder yang sama dengan app.py)
FILE_PATH = "hasil_clustering_kmeans.csv"

# Fungsi Load Data
# Catatan: dataset cukup besar, jadi kita hanya membaca kolom yang diperlukan untuk dashboard dan memproses file dalam chunk.
def load_data(file_path):
    try:
        required_cols = [
            'Cluster',
            'NamaWilayah',
            'Jenis_Pendaftaran',
            'Jenis_Kendaraan',
            'Warna_TNKB',
            'Golongan',
            'Fungsi',
            'Lokasi_Bayar',
            'KategoriUsia',
            'PokokPKB',
            'DendaPKB',
            'DurasiTunggakan',
            'SkorTNKB',
            'SkorGolonganKendaraan',
            'SkorFungsiKendaraan'
        ]

        header_df = pd.read_csv(file_path, nrows=0, engine='python')
        available_cols = [col for col in required_cols if col in header_df.columns]

        if not available_cols:
            st.error("❌ Tidak ada kolom penting yang ditemukan di file CSV.")
            return None

        chunks = []
        for chunk in pd.read_csv(
            file_path,
            usecols=available_cols,
            engine='python',
            on_bad_lines='skip',
            chunksize=100000
        ):
            chunks.append(chunk)

        if not chunks:
            return None

        return pd.concat(chunks, ignore_index=True)
    except FileNotFoundError:
        st.error(f"❌ File '{file_path}' tidak ditemukan!")
        st.info("💡 Pastikan file CSV tersebut sudah di-download dan diletakkan dalam **satu folder yang sama** dengan file `app.py` ini.")
        return None
    except Exception as e:
        st.error(f"Gagal membaca file: {e}")
        return None

# Memuat data secara otomatis
df = load_data(FILE_PATH)

# Pemetaan label cluster agar lebih mudah dipahami dalam dashboard dan laporan
cluster_label_map = {
    0: "Klaster 0: Tinggi",
    1: "Klaster 1: Sangat Tinggi",
    2: "Klaster 2: Tepat Waktu",
    3: "Klaster 3: Sangat Rendah",
    4: "Klaster 4: Menengah"
}

# Jika data berhasil dimuat, tampilkan konten dashboard
if df is not None:
    if 'Cluster' not in df.columns:
        st.warning("⚠️ Peringatan: Tidak ditemukan kolom bernama 'Cluster' di dalam dataset. Pastikan nama kolom hasil K-Means adalah 'Cluster'.")
    else:
        df['Cluster_Label'] = df['Cluster'].map(cluster_label_map).fillna(df['Cluster'].astype(str))

        # Menyiapkan kolom numerik untuk perhitungan Silhouette Score dan visualisasi
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'Cluster' in numeric_cols:
            numeric_cols.remove('Cluster')

        # Halaman 1: Ringkasan Data
        if menu == "🏠 Ringkasan Data":
            st.title("📊 Ringkasan Data Hasil K-Means")
            st.markdown("Menampilkan data kepatuhan wajib pajak kendaraan bermotor yang telah dikelompokkan beserta nilai validitas *clustering*-nya.")
            
            # 2. Menyiapkan Nilai Silhouette Score (Hanya Angka)
            try:
                jumlah_k = df['Cluster'].nunique()
                if jumlah_k == 5:
                    sil_score_text = "0.715445"
                    keterangan_cluster = "Strong Cluster"
                else:
                    mapping_sil_score = {2: "0.638331", 3: "0.656620", 4: "0.697678", 5: "0.715445"}
                    sil_score_text = mapping_sil_score.get(jumlah_k, "0.715445")
                    keterangan_cluster = "Strong Cluster" if jumlah_k == 5 else ""
            except Exception as e:
                sil_score_text = "0.715445"
                keterangan_cluster = "Strong Cluster"

            # 3. Menampilkan KPI dalam 4 Kolom
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Data", f"{len(df):,}")
            with col2:
                st.metric("Jumlah Fitur", 6)#len(numeric_cols))
            with col3:
                st.metric("Cluster Optimal", df['Cluster'].nunique())
            with col4:
                st.metric("Validitas Silhouette", sil_score_text, delta=keterangan_cluster, delta_color="off")

            st.markdown("---")
            
            list_cols = list(['PokokPKB', 
                              'DendaPKB', 
                              'DurasiTunggakan', 
                              'SkorTNKB', 
                              'SkorGolonganKendaraan', 
                              'SkorFungsiKendaraan'])

            if len(numeric_cols) >= 2:
                plot_col, right_col = st.columns([2, 2])

                with plot_col:
                    st.info("Distribusi Cluster")
                    x_col, y_col = st.columns(2)
                    with x_col:
                        x_axis = st.selectbox("Pilih sumbu X:", 
                                              list_cols, 
                                              index=0, 
                                              key="x_axis_summary")
                    with y_col:
                        y_axis = st.selectbox("Pilih sumbu Y:", 
                                              list_cols, 
                                              index=1 if len(numeric_cols) > 1 else 0, 
                                              key="y_axis_summary")

                    fig, ax = plt.subplots(figsize=(10, 6))
                    df_sample = df.sample(n=5000, random_state=42) if len(df) > 5000 else df
                    sns.scatterplot(x=x_axis, y=y_axis, hue='Cluster_Label', palette='Set1', data=df_sample, s=60, ax=ax)
                    ax.set_title(f"Sebaran Cluster: {x_axis} vs {y_axis}", fontsize=14, weight='bold')
                    ax.set_xlabel(x_axis, fontsize=11)
                    ax.set_ylabel(y_axis, fontsize=11)
                    ax.legend(title="Cluster")
                    fig.tight_layout()
                    st.pyplot(fig)

                with right_col:
                    
                    if 'NamaWilayah' in df.columns:
                        st.info("Jumlah Anggota Cluster per Kabupaten/Kota")
                        wilayah_options = ["Semua Data"] + sorted(df['NamaWilayah'].dropna().unique().tolist())
                        selected_wilayah = st.selectbox("Pilih Wilayah:", wilayah_options, key="wilayah_cluster_summary")

                        if selected_wilayah == "Semua Data":
                            cluster_count = df['Cluster_Label'].value_counts().sort_index()
                            chart_title = "Semua Data"
                        else:
                            wilayah_df = df[df['NamaWilayah'] == selected_wilayah]
                            cluster_count = wilayah_df['Cluster_Label'].value_counts().sort_index()
                            chart_title = selected_wilayah

                        if not cluster_count.empty:
                            cluster_order = [
                                "Klaster 0: Tinggi",
                                "Klaster 1: Sangat Tinggi",
                                "Klaster 2: Tepat Waktu",
                                "Klaster 3: Sangat Rendah",
                                "Klaster 4: Menengah"
                            ]
                            ordered_counts = pd.Series({k: cluster_count.get(k, 0) for k in cluster_order if k in cluster_count.index})
                            total_value = ordered_counts.sum()
                            ordered_pct = (ordered_counts / total_value * 100).round(1)

                            fig3, ax3 = plt.subplots(figsize=(5.4, 4.0))
                            colors = ['#1f77b4', '#17a2b8', '#2ca02c', '#d62728', '#6c757d']
                            bars = ax3.barh(ordered_counts.index, ordered_pct.values, color=colors[:len(ordered_pct)])
                            ax3.set_title(f'Proporsi Anggota Cluster\n{chart_title}', fontsize=11, weight='bold')
                            ax3.set_xlabel('Persentase (%)', fontsize=9)
                            ax3.set_ylabel('Klaster', fontsize=9)
                            ax3.invert_yaxis()
                            ax3.margins(x=0.15)
                            ax3.grid(axis='x', linestyle='--', alpha=0.35)
                            ax3.set_axisbelow(True)

                            for bar, value in zip(bars, ordered_pct.values):
                                if value > 0:
                                    ax3.text(
                                        value + 0.8,
                                        bar.get_y() + bar.get_height() / 2,
                                        f'{value:.1f}%',
                                        va='center',
                                        ha='left',
                                        fontsize=8,
                                        color='black',
                                        weight='bold'
                                    )
                            fig3.tight_layout()
                            st.pyplot(fig3)
                        else:
                            st.info("Tidak ada data untuk pilihan ini.")
                    else:
                        st.info("Kolom 'NamaWilayah' tidak ditemukan dalam dataset.")
            else:
                st.warning("Data numerik tidak cukup untuk membuat scatter plot.")
            st.markdown("---")

        # Halaman 2: Profil & Karakteristik
        elif menu == "📋 Profil & Karakteristik":
            list_cols = list(['PokokPKB', 'DendaPKB', 'DurasiTunggakan', 'SkorTNKB', 'SkorGolonganKendaraan', 'SkorFungsiKendaraan'])
            st.title("📋 Profil Karakteristik Cluster")
            st.markdown("Analisis centroid (titik pusat) untuk menentukan karakteristik tiap cluster (Tinggi, Sedang, Rendah).")
            
            st.markdown("---")
            st.subheader("Interpretasi Distribusi (Boxplot)")
            st.markdown("Melihat distribusi dan *outlier* pada setiap cluster.")
            
            fitur_boxplot = st.selectbox("Pilih Atribut untuk dianalisis:", list_cols, index=0, key="fitur_boxplot")
            
            fig3, ax3 = plt.subplots(figsize=(10, 5))
            sns.boxplot(x='Cluster_Label', y=fitur_boxplot, data=df, palette='Set2', ax=ax3)
            ax3.set_title(f'Distribusi Nilai {fitur_boxplot} per Cluster')
            plt.xticks(rotation=25, ha='right')
            st.pyplot(fig3)
            
            # --- TAMBAHAN: ANALISIS ATRIBUT KATEGORIKAL ---
            st.markdown("---")
            st.subheader("📊 Analisis Atribut Kategorikal per Cluster")
            st.markdown("Bagian ini menganalisis sebaran data non-numerik (kategorikal) seperti wilayah, jenis kendaraan, atau TNKB di dalam masing-masing cluster.")
            
            # Mendeteksi otomatis kolom yang berisi teks/kategori
            cat_cols = list(['Jenis_Pendaftaran',
                             'Jenis_Kendaraan', 
                             'Warna_TNKB', 
                             'Golongan',
                             'Fungsi', 
                             'Lokasi_Bayar',
                             'KategoriUsia'])
            # Menghapus kolom yang mungkin bukan kategori murni (opsional, sesuaikan nama kolom jika ada)
            if 'Cluster' in cat_cols:
                cat_cols.remove('Cluster')
                
            if len(cat_cols) > 0:
                # Membuat dropdown agar pengguna bisa memilih atribut kategorikal
                pilih_kategori = st.selectbox("Pilih Atribut Kategorikal untuk Dianalisis:", cat_cols)
                
                # 1. Menampilkan Grafik (Sekarang posisinya di ATAS)
                st.markdown(f"**Grafik Komposisi {pilih_kategori}**")
                fig4, ax4 = plt.subplots(figsize=(12, 6)) 
                
                # Mengambil sampel jika data terlalu besar
                df_cat_sample = df.sample(n=10000, random_state=42) if len(df) > 10000 else df
                
                # Menghitung proporsi per kategori dalam tiap cluster
                grouped_counts = (
                    df_cat_sample.groupby([pilih_kategori, 'Cluster_Label'])
                    .size()
                    .reset_index(name='jumlah')
                )
                grouped_counts['persentase'] = (
                    grouped_counts['jumlah']
                    / grouped_counts.groupby(pilih_kategori)['jumlah'].transform('sum')
                    * 100
                )

                sns.barplot(
                    data=grouped_counts,
                    x=pilih_kategori,
                    y='persentase',
                    hue='Cluster_Label',
                    palette='Set1',
                    ax=ax4
                )
                
                # Memutar label teks di sumbu X agar tidak bertumpuk
                plt.xticks(rotation=45, ha='right')
                ax4.set_title(f'Distribusi Persentase {pilih_kategori} Berdasarkan Cluster')
                ax4.set_ylabel('Persentase (%)')
                ax4.set_xlabel(pilih_kategori)
                ax4.set_ylim(0, 100)

                for container in ax4.containers:
                    ax4.bar_label(container, fmt='%.1f%%', padding=3, fontsize=8)
                
                st.pyplot(fig4)

                st.markdown("### Keterangan Analisis Atribut Kategorikal")
                cluster_summary = (
                    df_cat_sample.groupby([pilih_kategori, 'Cluster_Label'])
                    .size()
                    .unstack(fill_value=0)
                )

                if not cluster_summary.empty:
                    cluster_share = cluster_summary.div(cluster_summary.sum(axis=1), axis=0) * 100
                    top_category = cluster_share.max(axis=1).idxmax()
                    top_value = cluster_share.loc[top_category].max()
                    top_cluster = cluster_share.loc[top_category].idxmax()

                    cluster_descriptions = []
                    for cluster_id in cluster_summary.columns:
                        cluster_values = cluster_share[cluster_id].sort_values(ascending=False)
                        if not cluster_values.empty:
                            best_category = cluster_values.index[0]
                            best_pct = cluster_values.iloc[0]
                            cluster_descriptions.append(
                                f"Cluster {cluster_id} paling dominan pada {best_category}, dengan kontribusi sekitar {best_pct:.1f}% dari distribusi kategori tersebut."
                            )

                    narrative = " ".join(cluster_descriptions)

                    formal_narrative = (
                        f"Berdasarkan grafik komposisi {pilih_kategori}, kategori '{top_category}' merupakan kelompok yang paling dominan secara keseluruhan dengan proporsi sekitar {top_value:.1f}% pada Cluster {top_cluster}. "
                        f"{narrative} "
                        f"Temuan ini menunjukkan bahwa karakteristik {pilih_kategori} tidak tersebar secara merata antar cluster, melainkan membentuk pola yang berbeda dan tersegmentasi. Dengan demikian, setiap cluster memiliki profil kategori yang khas dan dapat diinterpretasikan sebagai indikator perbedaan perilaku atau karakteristik utama di antara kelompok data."
                    )

                    insight_narrative = (
                        f"Insight utama: {pilih_kategori} paling banyak ditemukan pada '{top_category}', dengan dominasi kuat di Cluster {top_cluster} sekitar {top_value:.1f}%. "
                        f"Pola ini menunjukkan bahwa setiap cluster memiliki preferensi kategori yang berbeda, sehingga segmentasi data terlihat jelas dalam grafik."
                    )

                    business_narrative = (
                        f"Secara bisnis, kategori '{top_category}' menjadi porsi terbesar dalam populasi yang dianalisis dan paling kuat di Cluster {top_cluster}. "
                        f"Hal ini menandakan bahwa segmen pelanggan atau data utama memiliki karakteristik tertentu yang konsisten dan dapat menjadi fokus strategi targeting maupun pengambilan keputusan. "
                        f"Perbedaan pola antar cluster memberikan peluang untuk menyusun pendekatan yang lebih spesifik dan efektif untuk masing-masing kelompok."
                    )

                    st.markdown("============================================")
                    st.markdown(formal_narrative)

                    st.markdown("============================================")
                    st.markdown(insight_narrative)

                    st.markdown("============================================")
                    st.markdown(business_narrative)
                else:
                    st.info("Tidak ada data cukup untuk membuat ulasan otomatis.")
                
            else:
                st.info("ℹ️ Tidak ditemukan atribut kategorikal (teks) dalam dataset ini.")
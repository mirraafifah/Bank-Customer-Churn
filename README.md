<h1>Bank Customer Churn Prediction </h1>
<h2>STAGE 1</h2>
<h3>BACKGROUND:</h3>

A bank is a financial institution that has influence on a country's economy. However, the competitive banking sector led to a increase in customer churn. Bank churn rates generally range from 5-10%. Increasing retention rates by as much as 5% can increase a bank's revenue by as much as 85%. Thus, customer reductions should be preventable in the way companies should invest more in maintaining relationships with customers.

Based on the existing dataset, our company has a churn rate by 20.37%. This number is quite high, so it is likely that customers to close the account or customer reductions will be quite a lot. So, an action is needed to predict the customer who will churn by creating machine learning models.

<h3>GOAL:</h3>

Lower the churn rate

###OBJECTIVES:

Build a classification model that can predict which customers will churn or not

##BUSINESS INSIGHT FROM EDA

1. Geography

Berdasarkan pengelompokan geografinya, Germany memiliki nilai churn paling besar dibanding France dan Spain yaitu 32,44%. Perlu dilakukan evaluasi pada nasabah Germany dikarenakan dengan Total Jumlah Nasabah yang lebih sedikit dibandingkan dengan France, namun Germany memiliki %Churn lebih tinggi.
Persentase Churn untuk France dan Spain hampir sama yaitu di 16%
2. Gender

Nasabah perempuan memiliki persentase Churn lebih besar daripada nasabah laki-laki yaitu 25.07%
3. Balance

Berdasarkan nilai yang mendominasi, balance paling banyak terjadi pada balance 0 dan rentang 100000 - 150000
Jika melihat distribusi balance per geography, Germany memiliki balance terbanyak pada rentang 100000 - 150000 sedangkan France dan Spain balance terbanyak pada balance 0.
4. Geography dan Balance

Berdasarkan perhitungan churn per Geography sebelumnya, didapatkan nilai churn paling besar pada Germany berkisar 32%. Pada grafik kde-plot, dapat dilihat adanya churn pada Nasabah Germany yang mempunyai balance 100000 - 15000. Seharusnya, nasabah yang mempunyai balance yang cukup besar tidak akan mempunyai keinginan untuk churn.
Dengan demikian, kita dapat simpulkan bahwa adanya problem pada nasabah Germany.
Berdasarkan perhitungan churn di France dan Spain, churn masih berkisar 16 %. Angka tersebut masih dianggap cukup besar untuk churn. Jika melihat balancenya, ternyata nasabah France dan Spain memiliki banyak nasabah yang memiliki saldo 0 sehingga memiliki kecenderungan untuk churn.
5. Number of Product

Dari analisis yang dilakukan, Nasabah paling banyak churn berdasarkan NumOfProduct yang dimiliki dari setiap masing- masing produk dan dari total keseluruhan NumofProduct adalah nasabah yang memiliki jumlah product di bank sebanyak 1. Dengan persentase 27,71% dan 14,09%.
Nasabah bank banyak menggunakan 2 produk dengan nilai churn yang kecil
Sehingga perlu dilakukan evaluasi dan treatment untuk Nasabah dengan 1 produk agar meningkatkan penggunaan produk nya menjadi 2, karena Nasabah dengan 2 produk churn cenderung kecil.
6. Age

Pada rentang usia 40 tahun hingga 50 tahun, jumlah Nasabah yang retained cenderung menurun, namun jumlah customer yang exited meningkat. Dengan kata lain pada rentang usia tersebut % churn meningkat.
Business Recommendation

1. Geography

Menganalisis lebih lanjut kondisi bisnis pada geography ‘Germany’ untuk memeriksa kemungkinan adanya kompetitor bisnis yang lebih baik sehingga dapat dilakukan benchmarking terhadap kompetitor.
Customers pada geography ‘Spain’ dan ‘France’ memiliki akun dengan balance 0 yang cukup tinggi, hal itu meningkatkan kecenderungan customers untuk churn. Sebagai rekomendasi, tim bisnis dapat menawarkan produk-produk rendah bunga dan minim deposit yang dapat menjangkau customers yang akan churn untuk mengaktifkan kembali akunnya, karena berdasarkan data, customer yang aktif memiliki kecenderungan untuk tidak ‘churn’.
2. Gender

Menawarkan produk-produk yang spesifik sesuai personality dan kebutuhan perempuan karena nasabah perempuan lebih cenderung untuk churn. seperti : Lady’s Card berupa kartu untuk nasabah perempuan yang mandiri secara finasial/ wanita karir, Produk investasi untuk perempuan yang sudah menikah/berkeluarga.
3.Number of Product

Meningkatkan Nasabah dari yang memiliki produk 1 menjadi produk 2 dengan cara promosi bundling produk.
4.Age

Memberikan promosi seperti diskon pembelian makan jika melakukan pembayaran bank tersebut dengan target nasabah usia produktif bekerja (40 tahun ke atas). Menawarkan produk persiapan pensiun dengan return yang tinggi kepada target nasabah usia 40-50 tahun.
##STAGE 2
##Data Pre-Processing
###A. Data Cleansing

A. Handle missing values

Dilakukan proses handle missing value dengan menggunakan df.isna().sum() pada dataset, dan data memiliki 14 kolom dengan tidak memiliki missing value.

B. Handle duplicated data

Tidak ditemukan data duplikat pada dataset dan jika hanya melihat kolom CustomerId dan Surname, juga tidak ditemukan data duplikat.

C. Feature Encoding

Mengubah feature categorical Gender dan Geography menggunakan one hot encoding (karena nominal bukan ordinal) menajadi feature numeric.

Feature Extraction

Sebelum melanjutkan data cleansing selajutnya, dilakukan proses feature engineering berupa Feature Extraction untuk menambahkan feature- feature yang ada dari feature yang sudah ada di dataset. Kami menambahkan 4 feature baru sebagai berikut :

Segment Age: Umur < 25 Umur 25 - <44 Umur 44 - <60 Umur 60 - < 75 Umur 75 ke atas

Segment Credit Score CreditScore < 580 CreditScote 580 - < 669 Credit Score 669 - < 739 CreditScore 739 - 779 CreditScore 779 ke atas

Segment Balance: Balance < 50000 Balance 50000 - < 100000 Balance 100000 - <150000 Balance 150000 ke atas

Segment EstimatedSalary: EstimatedSalary < 50000 EstimatedSalary 50000 - < 100000 EstimatedSalary 100000 - <150000 EstimatedSalary 150000 ke atas

D. Handling Outlier

Melakukan handling outlier pada feature age dengan menggunakan transformasi logaritma, dikarenakan apabila dilakukan analisis untuk distribusi age menunjukkan distribusi right skew. Sehingga setelah dilakukakan handling outlier menggunakan transformasi logaritma, terdapat perubahan distribusi pada kolom Log_age menjadi yang mendekati distribusi normal.

E. Feature Transformation

Normalisasi dilakukan pada kolom CreditScore, Balance, dan Estimated Salary sehingga didapatkan kolom yang berisi nilai 0 hingga 1 untuk memudahkan mesin dalam melakukan pemodelan dengan skala yang sama.

F. Handle Class Imbalance

Melakukan pemisahan untuk data train dan data test dengan komposisi 70% untuk training & 30% untuk testing. Dari hasil pemisahan data ini ytrain = 5574 dan ytest = 1426. Sehingga untuk dilakukan Handle Class Imbalance agar data tidak timpang, kami menggunakan SMOTE Oversampling. Hal ini disebabkan data yang kami miliki berjumlah sedikit. Sehingga diperoleh ytrain = 5574 dan ytest = 5574.

###B. Feature Engineering

A. Feature selection

A.1 Heatmap Correlation

Untuk mengetahui secara singkat hubungan antar fitur dan hubungan antara fitur dengan target, dilakukan mapping correlation
Berdasarkan heatmap, tidak ada multikolinearitas antar fitur (tidak ada fitur redundan)
Berdasarkan linear correlation, fitur HasCrCard, Tenure, EstimatedSalary memiliki skor paling rendah terhadap fitur target
Diperlukan analisis fitur lebih lanjut karena heatmap corr hanya melihat hubungan linear corr. sedangkan fitur yang ada merupakan campuran numerik dan kategorik
Analisis Korelasi Non Linear

A.2 Mutual Information Score
Berdasarkan mutual information score didapat 7 feature yang paling berpengaruh adalah NumofProducts, Age, isActiveMember, Geography, Balance, dan Gender.
HasCrCard dan CreditScore memiliki nilai mutual information yang rendah.
HasCrCard memiliki score yang sangat rendah, dapat dipertimbangkan untuk menghilangkan feature ini dari pemodelan.
A.3 Chi-square Test
Berdasarkan chi-square dan p-value score didapat 7 feature yang berpengaruh dari yang paling tinggi adalah Balance, EstimatedSalary, Age, IsActiveMember, CreditScore, Mapping_Geography, Mapping_Gender.
HasCrCard, Tenure, NumOfProduct memiliki nilai chi-square yang rendah dan p-value yang lebih tinggi dibandingkan feature lainnya.
HasCrCard memiliki score yang sangat rendah, dapat dipertimbangkan untuk menghilangkan feature ini dari pemodelan.
A.4 ANOVA
Berdasarkan F scores dan p-value score didapat feature yang berpengaruh adalah Age, IsActiveMember, Mapping_Geography, Balance, Mapping_Gender, NumOfProducts, CreditScore
HasCrCard, EstimatedSalary, Tenure memiliki nilai F score yang rendah dan p-value yang lebih tinggi dibandingkan feature lainnya.
HasCrCard memiliki score yang sangat rendah, dapat dipertimbangkan untuk menghilangkan feature ini dari pemodelan.
A.5 Feature Importance
Berdasarkan Feature Importance score didapat 7 feature yang berpengaruh dari yang paling tinggi adalah Age, IsActiveMember, CreditScore, Gender, Balance, Geography, EstimatedSalary,
NumOfProducts, Tenure, dan HasCrCard memiliki nilai feature importance yang rendah.
HasCrCard memiliki score yang sangat rendah, dapat dipertimbangkan untuk menghilangkan feature ini dari pemodelan.
Berdasarkan analisis korelasi linear dan non linear diatas dapat kami simpulkan bahwa Fitur yang terseleksi=

RowNumber, CustomerID, Surname karena informasinya tidak diperlukan.
Segment Age, Segment CreditScore, Segment Balance, Segment EstimatedSalary karena data asli memiliki nilai yang lebih tinggi.
HasCrCard karena memiliki value yang rendah.

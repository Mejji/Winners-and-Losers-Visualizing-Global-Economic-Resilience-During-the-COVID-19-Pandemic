# Winners and Losers: Visualizing Global Economic Resilience During the COVID-19 Pandemic

Authors:
Alvear, Mark Josh 
Jayin, Aaron James 
Tolentino, John Carl 


## 📌 Overview
This project analyzes **global economic resilience** from 2010 to 2023, with a special focus on the **COVID-19 pandemic (2020–2022)**. Using macroeconomic indicators from the **World Bank World Development Indicators (WDI)**, supplemented by IMF and OECD data, the study compares nearly 200 nations to understand how economies absorbed shocks and recovered growth.

The project contributes to **Sustainable Development Goals (SDG 1: No Poverty, SDG 8: Decent Work and Economic Growth)** by providing accessible, empirical, and visual insights into resilience patterns worldwide.

---

## 🎯 Objectives
1. **Track resilience** to external shocks (2010–2025), measuring contraction depth and recovery speed.  
2. **Categorize nations** into resilience profiles using indicators like GDP growth, unemployment, inflation, and fiscal stability.  
3. **Highlight regional trends** and case studies of both vulnerability and strength.  

---

## 📊 Dataset
- **Source:** World Bank WDI, IMF, OECD  
- **Coverage:** ~200 countries, 2010–2025  
- **Indicators:**
  - Public Debt (% of GDP)  
  - GDP Growth (% Annual)  
  - GDP per Capita (USD)  
  - Gross National Income (USD)  
  - Inflation (CPI %, GDP Deflator %)  
  - Unemployment Rate (%)  
  - Government Revenue & Expense (% of GDP)  
  - Current Account Balance (% of GDP)  
  - Real Interest Rate (%)  

---

## 🛠️ Methodology
- **Preprocessing:**  
  - Missing values handled via interpolation  
  - Normalization (Min-Max scaling) for comparability  
  - Segmentation into **Pre-pandemic (2010–2019)**, **Pandemic (2020–2021)**, and **Recovery (2022–2025)** phases  

- **Resilience Index Construction:**  
  Composite score based on weighted dimensions:  
  

\[
  \text{Resilience Index} = (Fiscal \times 0.3) + (Growth \times 0.3) + (External \times 0.2) + (Monetary \times 0.2)
  \]



- **Visualization Tools:**  
  - Choropleth maps (global resilience distribution)  
  - Time series (GDP, inflation, debt trends)  
  - Ridgeline plots (distribution shifts)  
  - Scatterplots & cluster analysis (resilient vs fragile economies)  
  - Interactive dashboards (Tableau / Plotly)  

---

## 📈 Key Findings
- Nations with **low public debt and stable inflation** absorbed shocks better.  
- **Switzerland, Macao SAR, Djibouti, Guyana, Papua New Guinea** ranked among the most resilient in 2023.  
- The **Global North** generally recovered faster, while parts of Africa, South Asia, and Latin America remained fragile.  
- Resilience disparities widened during the pandemic, highlighting structural inequalities.  

---

## 🚀 Usage
Clone the repository and explore the visualizations:

```bash
git clone https://github.com/your-username/global-economic-resilience.git
cd global-economic-resilience

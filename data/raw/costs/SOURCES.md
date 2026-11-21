# Facility cost sources (Rs crore)

`facility_costs.csv` lists, for each facility type, the capital cost and its source, and the annual operating cost and its basis.

- Health operating costs come from Prinja et al. (2016), PLoS ONE 11(8): e0160986 (2012-13 prices), uprated by the All-India CPI
  (Combined, base 2012 = 100; January 2025 = 193.5, MoSPI).
- Fire station: DFES revenue budget 2025-26 (Rs 91.87 crore, Goa Budget Vol. III, Demand No. 26) divided by 17 units (the 16 fire stations listed on the DFES organisation page plus the Panaji headquarters).
- Degree college: Goa Budget 2025-26 Vol. II, Demand No. 35 (Higher Education), head 2202-03-103-01 "Government College",
  budget estimate Rs 9,767.75 lakh (salaries Rs 8,200 lakh), divided by the 7 institutions in the Directorate of Higher
  Education's "Government Colleges" list (dhe.goa.gov.in/institute/getinstitut/).
- Stadium: Sports Authority of Goa statement that it spends "around Rs one crore" a year on maintaining the Campal indoor
  stadium (Herald Goa, "SAG decides to lease Shyama Prasad indoor stadium to private operators").
- Public library: Directorate of Art & Culture, Performance Budget 2025-26, head 2205-105-06 (four government village
  libraries), 2024-25 actual expenditure Rs 114.43 lakh, i.e. Rs 0.286 crore per library.
- Auditorium / community hall: no published Goa figure; Rs 0.05 crore a year is an assumption.

Every cost is varied in the Monte Carlo (log-normal, sigma = 0.4).

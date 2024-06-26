# DataAnalysisProgram

Курсовой проект по дисциплине: "Вычислительная математика"  
Разработка данного программного обеспечения (ПО) проходила на втором курсе обучения в ВУЗе.  
Тема проекта: "Визуализация данных"


Постановка задачи:  
Вычислить наименьший положительный корень T уравнения: x^5-7x-10=0  
Положение корня локализовать методом прямого поиска с шагом 1. Для уточнения величины T применить метод касательных. На отрезке [0,2T] проинтегрировать дифференциальное уравнение: Y'=0.1*t^2-2tY(t)+(1+(n/10))*Fs(t,τ)  
с начальными условиями Y(0)=0.2, где n – последняя цифра студенческого билета, Fs(t,τ) – периодическая функция с периодом τ=T, имеющая вид на основном отрезке периодичности [0,T]:
Fs(t,τ)={(1,0<t<1,(τ-t)/(τ-1),1≤t≤τ.)  
Для интегрирования применить метод Эйлера второго порядка с коррекцией по средней производной.
Шаг интегрирования задать как h=T/N, где 2N+1 – число дискретных точек на интервале интегрирования, N∈[15,30].
	Полученные дискретные значения функции Y(t) аппроксимировать интерполяционным полиномом в форме Лагранжа степени не ниже третьей. Для этого отобрать соответствующее число точек интерполяции, равномерно расположенных на отрезке [0,2T].
	Для всех вычисленных дискретных значений функции Y(t) построить многочлен (той же степени, что и интерполяционный полином Лагранжа), аппроксимирующий табличную функцию по методу наименьших квадратов. Для вычисления коэффициентов полинома построить процедуру, решающую систему линейных уравнений с симметричной матрицей методом квадратного корня (метод Холецкого разложения матрицы на произведение треугольных матриц).
Построить графики сеточной функции и аппроксимирующих её полиномов. 
Решить задачу встроенными процедурами системы Maple и сравнить результаты интегрирования дифференциального уравнения.

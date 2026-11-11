"""
Pruebas automáticas de la aplicación (caso normal, entradas inválidas y registros conocidos).

Ejecutar desde la carpeta del proyecto:
    python pruebas_app.py
"""

from streamlit.testing.v1 import AppTest


def llenar_formulario(at, **valores):
    sexo = valores.pop("sexo", "Femenino")
    no_disponibles = valores.pop("no_disponible", [])
    at.selectbox[0].set_value(sexo)
    for variable, valor in valores.items():
        at.number_input(key=f"campo_{variable}").set_value(valor)
    for variable in ["bloodpressure", "skinthickness", "insulin"]:
        at.checkbox(key=f"nd_{variable}").set_value(variable in no_disponibles)
    at.button[0].click()
    return at.run(timeout=60)


def probabilidad_mostrada(at):
    return [m.value for m in at.metric if m.label == "Probabilidad estimada de diabetes"]


def iniciar():
    at = AppTest.from_file("app.py", default_timeout=60).run()
    assert not at.exception, at.exception
    return at


resultados = []


def registrar(nombre, ok, detalle):
    resultados.append((nombre, ok, detalle))
    print(f"[{'OK' if ok else 'FALLA'}] {nombre}: {detalle}")


# 1. Caso normal (mismo caso de la sección 15.7 del notebook: probabilidad 0.267, clase 0)
at = llenar_formulario(iniciar(), age=30, pregnancies=2, glucose=110.0, bmi=27.5,
                       diabetes_pedigree_function=0.35, bloodpressure=70.0, skinthickness=25.0, insulin=100.0)
prob = probabilidad_mostrada(at)
clase = [s.value for s in at.success if "Clase predicha" in s.value]
registrar("Caso normal", prob == ["26.7%"] and len(clase) == 1 and "0 – No diabetes" in clase[0] and not at.error,
          f"probabilidad {prob}, {clase[0][2:34] if clase else 'sin clase'}")

# 2. Datos no disponibles (notebook: probabilidad 0.918, clase 1)
at = llenar_formulario(iniciar(), age=45, pregnancies=5, glucose=165.0, bmi=36.0,
                       diabetes_pedigree_function=0.60, bloodpressure=76.0,
                       no_disponible=["skinthickness", "insulin"])
prob = probabilidad_mostrada(at)
registrar("Datos no disponibles (pliegue e insulina)", prob == ["91.8%"] and len(at.info) >= 1,
          f"probabilidad {prob}, aviso: {at.info[-1].value[:70] if at.info else '-'}")

# 3. Entrada inválida: edad 15, glucosa 0 e insulina 0 sin marcar «No disponible»
at = llenar_formulario(iniciar(), age=15, pregnancies=0, glucose=0.0, bmi=22.0,
                       diabetes_pedigree_function=0.3, bloodpressure=70.0, skinthickness=20.0, insulin=0.0)
texto_error = at.error[0].value if at.error else ""
registrar("Entrada inválida (edad, glucosa e insulina)",
          not probabilidad_mostrada(at) and all(t in texto_error for t in ["Edad", "Glucosa", "Insulina", "No disponible"]),
          "sin predicción; " + " | ".join(l.strip("- ") for l in texto_error.splitlines() if l.startswith("-")))

# 4. Categoría inválida: sexo masculino
at = llenar_formulario(iniciar(), sexo="Masculino", age=40, pregnancies=0, glucose=120.0, bmi=28.0,
                       diabetes_pedigree_function=0.4, bloodpressure=80.0, skinthickness=25.0, insulin=100.0)
texto_error = at.error[0].value if at.error else ""
registrar("Categoría inválida (sexo masculino)", not probabilidad_mostrada(at) and "solo con mujeres" in texto_error,
          "sin predicción; " + texto_error.splitlines()[-1].strip("- ") if texto_error else "sin error")

# 5. Valor fuera del rango de entrenamiento: advertencia, pero sí predice
at = llenar_formulario(iniciar(), age=35, pregnancies=1, glucose=250.0, bmi=30.0,
                       diabetes_pedigree_function=0.5, bloodpressure=80.0, skinthickness=25.0, insulin=150.0)
registrar("Fuera del rango de entrenamiento (glucosa 250)",
          bool(probabilidad_mostrada(at)) and any("rango observado" in w.value for w in at.warning),
          f"probabilidad {probabilidad_mostrada(at)} con advertencia")

# 6. Registro conocido de prueba, fila 408 (notebook: probabilidad 0.725, clase 1, valor real 1).
#    Pliegue e insulina valen 0 en el dataset (faltantes), así que se marcan como «No disponible».
at = llenar_formulario(iniciar(), age=39, pregnancies=8, glucose=197.0, bmi=25.9,
                       diabetes_pedigree_function=1.191, bloodpressure=74.0,
                       no_disponible=["skinthickness", "insulin"])
registrar("Registro de prueba fila 408 en el formulario", probabilidad_mostrada(at) == ["72.5%"],
          f"probabilidad {probabilidad_mostrada(at)} (notebook 72.5 %)")

# 7. Registro conocido de prueba, fila 44 (notebook: probabilidad 0.732, clase 1, valor real 0 → falsa alarma)
at = llenar_formulario(iniciar(), age=40, pregnancies=7, glucose=159.0, bmi=27.4,
                       diabetes_pedigree_function=0.294, bloodpressure=64.0,
                       no_disponible=["skinthickness", "insulin"])
registrar("Registro de prueba fila 44 en el formulario", probabilidad_mostrada(at) == ["73.2%"],
          f"probabilidad {probabilidad_mostrada(at)} (notebook 73.2 %)")

# 8. Pestaña de verificación: los 154 registros de prueba coinciden con el notebook
at = iniciar()
valor = [m.value for m in at.metric if m.label.startswith("Registros cuya predicción")]
registrar("Verificación de los 154 registros de prueba", valor == ["154 de 154"] and len(at.success) >= 1, f"{valor}")

print(f"\n{sum(ok for _, ok, _ in resultados)} de {len(resultados)} pruebas superadas")
if not all(ok for _, ok, _ in resultados):
    raise SystemExit(1)

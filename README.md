# Codigo fuente de las Diapositivas de la Presentación

Aqui se encuentra la presentacion de las diapositivas del proyecto de teoria de grafos, hechas con manim.

## Requisitos

Para ejecutar el codigo, necesitas tener instalado Manim. Puedes instalarlo siguiendo las instrucciones en la [documentación oficial de Manim](https://docs.manim.community/en/stable/installation.html).

## Instrucciones para ejecutar

1. Clona este repositorio en tu máquina local.
2. Navega al directorio del proyecto.
3. Ejecuta el siguiente comando para renderizar las diapositivas:
```
manim -pq[calidad} main.py 
```
Reemplaza `[calidad]` con `l` para baja calidad, `m` para media calidad, o `h` para alta calidad

Esto generara un video con las diapositivas.

Para volverlas diapositivas en formato html, con el siguiente comando

```
manim-slides convert Diapositivas.html --one-file --offline -ccontrols=true --open 
```

Eso le pedira seleccionar que archivo de diapositivas quiere convertir, seleccionar "DiapositivasProyecto"

De todas formas, en el repositorio esta incluido DiapositivasFinal.html que es el resultado final.

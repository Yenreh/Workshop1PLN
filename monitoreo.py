import time
import shutil
import subprocess
import os
import psutil  # Necesario para monitorear RAM

def gb_format(b):
    return f"{b / (1024**3):.2f} GB"

print("Iniciando monitoreo de GPU, RAM y Disco C: ...\n")

try:
    while True:
        # === MONITOREO DE GPU ===
        print("📊 Estado de la GPU:")
        try:
            result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                # Mostrar líneas relevantes (ajusta según tu salida de nvidia-smi)
                for line in lines[7:11]:
                    print(f"  {line}")
            else:
                print("  GPU: No disponible o no NVIDIA")
        except Exception as e:
            print(f"  Error al leer GPU: {e}")

        # === MONITOREO DE RAM ===
        print("\n🧠 Estado de la RAM:")
        try:
            mem = psutil.virtual_memory()
            print(f"  Total: {gb_format(mem.total)}")
            print(f"  Usada: {gb_format(mem.used)} ({mem.percent}%)")
            print(f"  Libre: {gb_format(mem.available)}")
        except Exception as e:
            print(f"  Error al leer RAM: {e}")

        # === MONITOREO DE DISCO C: ===
        print("\n💾 Estado del Disco:")
        if os.name == 'nt':  # Windows
            try:
                total, used, free = shutil.disk_usage("C:\\")
                print(f"  Unidad: C:")
                print(f"  Total: {gb_format(total)}")
                print(f"  Usado: {gb_format(used)}")
                print(f"  Libre: {gb_format(free)}")
            except Exception as e:
                print(f"  Error al leer disco C: {e}")
        else:
            # En Linux/macOS, monitorear la raíz
            try:
                total, used, free = shutil.disk_usage("/")
                print(f"  Unidad: / (raíz)")
                print(f"  Total: {gb_format(total)}")
                print(f"  Usado: {gb_format(used)}")
                print(f"  Libre: {gb_format(free)}")
            except Exception as e:
                print(f"  Error al leer disco raíz: {e}")

        print("\n" + "="*60 + "\n")
        time.sleep(3)

except KeyboardInterrupt:
    print("Monitoreo detenido por el usuario.")
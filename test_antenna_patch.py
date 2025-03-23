from antennas.rectangular_patch import RectangularPatch
import plotly.graph_objects as go

def main():
    # Cria uma instância da antena RectangularPatch
    antenna = RectangularPatch(
        fo=2.45e9,  # Frequência de operação em Hz
        patch_width=32,  # Largura do patch em mm
        patch_length=40,  # Comprimento do patch em mm
        substrate_thickness=1.524,  # Espessura do substrato em mm
        substrate_width=60,  # Largura do substrato em mm
        substrate_length=60,  # Comprimento do substrato em mm
        metal_thickness=0.05,  # Espessura do metal em mm
        feed_pos=-6.0  # Posição do feed em mm
    )

    # Salva a geometria em um arquivo .FCStd (formato do FreeCAD)
    #antenna.save_to_freecad(filename="rectangular_patch.FCStd")
    #print("✅ Arquivo FreeCAD salvo: rectangular_patch.FCStd")

    # Visualiza a geometria usando Plotly
    print("Visualizando a antena com Plotly...")
    antenna.visualize()

if __name__ == "__main__":
    main()
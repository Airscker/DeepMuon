'''
Author: airscker
Date: 2023-07-29 15:57:32
LastEditors: airscker
LastEditTime: 2023-07-30 12:33:39
Description: NULL

Copyright (C) 2023 by Airscker(Yufeng), All Rights Reserved. 
'''
from typing import Any,Union

class ElementPeriodTable:
    def __init__(self) -> None:
        self.__web_prefix='https://en.wikipedia.org/wiki/'
        self.__table={'H':[1, 1.008, 1, 'Hydrogen', 's', 'Nonmetal Gas'],
                    'He':[2, 4.0026, 1, 'Helium', 's', 'Noble Gas'],
                    'Li':[3, 6.94, 1, 'Lithium', 's', 'Alkali Solid'],
                    'Be':[4, 9.0122, 1, 'Beryllium', 's', 'Alkaline Solid'],
                    'B':[5, 10.81, 1, 'Boron', 'p', 'Metalloid Solid'],
                    'C':[6, 12.011, 1, 'Carbon', 'p', 'Nonmetal Solid'],
                    'N':[7, 14.007, 1, 'Nitrogen', 'p', 'Nonmetal Gas'],
                    'O':[8, 15.999, 1, 'Oxygen', 'p', 'Nonmetal Gas'],
                    'F':[9, 18.998, 1, 'Fluorine', 'p', 'Nonmetal Gas'],
                    'Ne':[10, 20.18, 1, 'Neon', 'p', 'Noble Gas'],
                    'Na':[11, 22.99, 1, 'Sodium', 's', 'Alkali Solid'],
                    'Mg':[12, 24.305, 1, 'Magnesium', 's', 'Alkaline Solid'],
                    'Al':[13, 26.982, 1, 'Aluminium', 'p', 'Poor Solid'],
                    'Si':[14, 28.085, 1, 'Silicon', 'p', 'Metalloid Solid'],
                    'P':[15, 30.974, 1, 'Phosphorus', 'p', 'Nonmetal Solid'],
                    'S':[16, 32.06, 1, 'Sulfur', 'p', 'Nonmetal Solid'],
                    'Cl':[17, 35.45, 1, 'Chlorine', 'p', 'Nonmetal Gas'],
                    'Ar':[18, 39.948, 1, 'Argon', 'p', 'Noble Gas'],
                    'K':[19, 39.098, 1, 'Potassium', 's', 'Alkali Solid'],
                    'Ca':[20, 40.078, 1, 'Calcium', 's', 'Alkaline Solid'],
                    'Sc':[21, 44.956, 1, 'Scandium', 'd', 'Transition Solid'],
                    'Ti':[22, 47.867, 1, 'Titanium', 'd', 'Transition Solid'],
                    'V':[23, 50.942, 1, 'Vanadium', 'd', 'Transition Solid'],
                    'Cr':[24, 51.996, 1, 'Chromium', 'd', 'Transition Solid'],
                    'Mn':[25, 54.938, 1, 'Manganese', 'd', 'Transition Solid'],
                    'Fe':[26, 55.845, 1, 'Iron', 'd', 'Transition Solid'],
                    'Co':[27, 58.933, 1, 'Cobalt', 'd', 'Transition Solid'],
                    'Ni':[28, 58.693, 1, 'Nickel', 'd', 'Transition Solid'],
                    'Cu':[29, 63.546, 1, 'Copper', 'd', 'Transition Solid'],
                    'Zn':[30, 65.38, 1, 'Zinc', 'd', 'Transition Solid'],
                    'Ga':[31, 69.723, 1, 'Gallium', 'p', 'Poor Solid'],
                    'Ge':[32, 72.63, 1, 'Germanium', 'p', 'Metalloid Solid'],
                    'As':[33, 74.922, 1, 'Arsenic', 'p', 'Metalloid Solid'],
                    'Se':[34, 78.971, 1, 'Selenium', 'p', 'Nonmetal Solid'],
                    'Br':[35, 79.904, 1, 'Bromine', 'p', 'Nonmetal Liquid'],
                    'Kr':[36, 83.798, 1, 'Krypton', 'p', 'Noble Gas'],
                    'Rb':[37, 85.468, 1, 'Rubidium', 's', 'Alkali Solid'],
                    'Sr':[38, 87.62, 1, 'Strontium', 's', 'Alkaline Solid'],
                    'Y':[39, 88.906, 1, 'Yttrium', 'd', 'Transition Solid'],
                    'Zr':[40, 91.224, 1, 'Zirconium', 'd', 'Transition Solid'],
                    'Nb':[41, 92.906, 1, 'Niobium', 'd', 'Transition Solid'],
                    'Mo':[42, 95.95, 1, 'Molybdenum', 'd', 'Transition Solid'],
                    'Tc':[43, 98.0, 0, 'Technetium', 'd', 'Transition Solid'],
                    'Ru':[44, 101.07, 1, 'Ruthenium', 'd', 'Transition Solid'],
                    'Rh':[45, 102.91, 1, 'Rhodium', 'd', 'Transition Solid'],
                    'Pd':[46, 106.42, 1, 'Palladium', 'd', 'Transition Solid'],
                    'Ag':[47, 107.87, 1, 'Silver', 'd', 'Transition Solid'],
                    'Cd':[48, 112.41, 1, 'Cadmium', 'd', 'Transition Solid'],
                    'In':[49, 114.82, 1, 'Indium', 'p', 'Poor Solid'],
                    'Sn':[50, 118.71, 1, 'Tin', 'p', 'Poor Solid'],
                    'Sb':[51, 121.76, 1, 'Antimony', 'p', 'Metalloid Solid'],
                    'Te':[52, 127.6, 1, 'Tellurium', 'p', 'Metalloid Solid'],
                    'I':[53, 126.9, 1, 'Iodine', 'p', 'Nonmetal Solid'],
                    'Xe':[54, 131.29, 1, 'Xenon', 'p', 'Noble Gas'],
                    'Cs':[55, 132.91, 1, 'Caesium', 's', 'Alkali Solid'],
                    'Ba':[56, 137.33, 1, 'Barium', 's', 'Alkaline Solid'],
                    'La':[57, 138.91, 1, 'Lanthanum', 'f', 'Lanthanoid Solid'],
                    'Ce':[58, 140.12, 1, 'Cerium', 'f', 'Lanthanoid Solid'],
                    'Pr':[59, 140.91, 1, 'Praseodymium', 'f', 'Lanthanoid Solid'],
                    'Nd':[60, 144.24, 1, 'Neodymium', 'f', 'Lanthanoid Solid'],
                    'Pm':[61, 145.0, 0, 'Promethium', 'f', 'Lanthanoid Solid'],
                    'Sm':[62, 150.36, 1, 'Samarium', 'f', 'Lanthanoid Solid'],
                    'Eu':[63, 151.96, 1, 'Europium', 'f', 'Lanthanoid Solid'],
                    'Gd':[64, 157.25, 1, 'Gadolinium', 'f', 'Lanthanoid Solid'],
                    'Tb':[65, 158.93, 1, 'Terbium', 'f', 'Lanthanoid Solid'],
                    'Dy':[66, 162.5, 1, 'Dysprosium', 'f', 'Lanthanoid Solid'],
                    'Ho':[67, 164.93, 1, 'Holmium', 'f', 'Lanthanoid Solid'],
                    'Er':[68, 167.26, 1, 'Erbium', 'f', 'Lanthanoid Solid'],
                    'Tm':[69, 168.93, 1, 'Thulium', 'f', 'Lanthanoid Solid'],
                    'Yb':[70, 173.05, 1, 'Ytterbium', 'f', 'Lanthanoid Solid'],
                    'Lu':[71, 174.97, 1, 'Lutetium', 'd', 'LanthanoiSolid'],
                    'Hf':[72, 178.49, 1, 'Hafnium', 'd', 'Transition Solid'],
                    'Ta':[73, 180.95, 1, 'Tantalum', 'd', 'Transition Solid'],
                    'W':[74, 183.84, 1, 'Tungsten', 'd', 'Transition Solid'],
                    'Re':[75, 186.21, 1, 'Rhenium', 'd', 'Transition Solid'],
                    'Os':[76, 190.23, 1, 'Osmium', 'd', 'Transition Solid'],
                    'Ir':[77, 192.22, 1, 'Iridium', 'd', 'Transition Solid'],
                    'Pt':[78, 195.08, 1, 'Platinum', 'd', 'Transition Solid'],
                    'Au':[79, 196.97, 1, 'Gold', 'd', 'Transition Solid'],
                    'Hg':[80, 200.59, 1, 'Mercury', 'd', 'Transition Liquid'],
                    'Tl':[81, 204.38, 1, 'Thallium', 'p', 'Poor Solid'],
                    'Pb':[82, 207.2, 1, 'Lead', 'p', 'Poor Solid'],
                    'Bi':[83, 208.98, 1, 'Bismuth', 'p', 'Poor Solid'],
                    'Po':[84, 209.0, 0, 'Polonium', 'p', 'Poor Solid'],
                    'At':[85, 210.0, 0, 'Astatine', 'p', 'Metalloid Solid'],
                    'Rn':[86, 222.0, 0, 'Radon', 'p', 'Noble Gas'],
                    'Fr':[87, 223.0, 0, 'Francium', 's', 'Alkali Solid'],
                    'Ra':[88, 226.0, 0, 'Radium', 's', 'Alkaline Solid'],
                    'Ac':[89, 227.0, 0, 'Actinium', 'f', 'Actinoid Solid'],
                    'Th':[90, 232.04, 1, 'Thorium', 'f', 'Actinoid Solid'],
                    'Pa':[91, 231.04, 1, 'Protactinium', 'f', 'Actinoid Solid'],
                    'U':[92, 238.03, 1, 'Uranium', 'f', 'Actinoid Solid'],
                    'Np':[93, 237.0, 0, 'Neptunium', 'f', 'Actinoid Solid'],
                    'Pu':[94, 244.0, 0, 'Plutonium', 'f', 'Actinoid Solid'],
                    'Am':[95, 243.0, 0, 'Americium', 'f', 'Actinoid Solid'],
                    'Cm':[96, 247.0, 0, 'Curium', 'f', 'Actinoid Solid'],
                    'Bk':[97, 247.0, 0, 'Berkelium', 'f', 'Actinoid Solid'],
                    'Cf':[98, 251.0, 0, 'Californium', 'f', 'Actinoid Solid'],
                    'Es':[99, 252.0, 0, 'Einsteinium', 'f', 'Actinoid Solid'],
                    'Fm':[100, 257.0, 0, 'Fermium', 'f', 'Actinoid Solid'],
                    'Md':[101, 258.0, 0, 'Mendelevium', 'f', 'Actinoid Solid'],
                    'No':[102, 259.0, 0, 'Nobelium', 'f', 'Actinoid Solid'],
                    'Lr':[103, 266.0, 0, 'Lawrencium', 'd', 'ActinoiSolid'],
                    'Rf':[104, 267.0, 0, 'Rutherfordium', 'd', 'Transition UnknownState'],
                    'Db':[105, 268.0, 0, 'Dubnium', 'd', 'Transition UnknownState'],
                    'Sg':[106, 269.0, 0, 'Seaborgium', 'd', 'Transition UnknownState'],
                    'Bh':[107, 270.0, 0, 'Bohrium', 'd', 'Transition UnknownState'],
                    'Hs':[108, 277.0, 0, 'Hassium', 'd', 'Transition UnknownState'],
                    'Mt':[109, 278.0, 0, 'Meitnerium', 'd', 'Unknown UnknownState'],
                    'Ds':[110, 281.0, 0, 'Darmstadtium', 'd', 'Unknown UnknownState'],
                    'Rg':[111, 282.0, 0, 'Roentgenium', 'd', 'Unknown UnknownState'],
                    'Cn':[112, 285.0, 0, 'Copernicium', 'd', 'Unknown UnknownState'],
                    'Nh':[113, 286.0, 0, 'Nihonium', 'p', 'Unknown UnknownState'],
                    'Fl':[114, 289.0, 0, 'Flerovium', 'p', 'Unknown UnknownState'],
                    'Mc':[115, 290.0, 0, 'Moscovium', 'p', 'Unknown UnknownState'],
                    'Lv':[116, 293.0, 0, 'Livermorium', 'p', 'Unknown UnknownState'],
                    'Ts':[117, 294.0, 0, 'Tennessine', 'p', 'Unknown UnknownState'],
                    'Og':[118, 294.0, 0, 'Oganesson', 'p', 'Unknown UnknownState'],
        }
    class ElementError(Exception):
        pass
    def reverse(self):
        reversed_table={}
        for key in self.__table.keys():
            reversed_table[self.__table[key]]=key
        return reversed_table
    def __call__(self, Element:Union[str,list,tuple]) -> Any:
        if isinstance(Element,str):
            if Element not in self.__table.keys():
                raise self.ElementError(f"Element {Element} doesn't exist on the earth.")
            else:
                return self.__table[Element]
        if isinstance(Element,list) or isinstance(Element,tuple):
            for i in range(len(Element)):
                if Element not in self.__table.keys():
                    raise self.ElementError(f"Element {Element[i]} doesn't exist on the earth.")
                else:
                    return self.__table[Element[i]]
    @property
    def AvailableElements(self):
        return list(self.__table.keys())

demo=ElementPeriodTable()
print(demo.AvailableElements)
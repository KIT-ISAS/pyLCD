# pyLCD

`pyLCD` is a Python library for generating samples that minimize the modified Cramér–von Mises distance for standard Gaussian distributions based on the localized cumulative distribution (LCD). These LCD samples can be transformed to match the mean and covariance of non-standard Gaussian distributions.
Currently, only symmetric LCD sample sets for even numbers of samples for standard Gaussian distributions can be generated. These samples can be linearly transformed to match the mean and covariance of non-standard Gaussian distributions.

## Usage

To use the library, refer to the `__main__` section of the `lcd_sym_even.py` file, which contains an example of how to generate symmetric LCD sample sets for even numbers of samples.

## License

This project is licensed under the MIT License.

## Author

Florian Pfaff, pfaff@kit.edu


## Further Reading

This work is based on the research and development of methods to minimize the modified Cramér-von Mises distance by Uwe D. Hanebeck, Jannik Steinbring, and Martin Pander.
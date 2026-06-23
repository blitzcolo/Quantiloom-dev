#include "ModtranCalc.hpp"
#include <toml++/toml.hpp>
#include <filesystem>
#include <iostream>

int main(int argc, char* argv[])
{
    const char* cfgPath = (argc > 1) ? argv[1] : "qltrans.toml";
    toml::table cfg;
    try { cfg = toml::parse_file(cfgPath); }
    catch (const toml::parse_error& e) {
        std::cerr << "Config parse error: " << e.description() << "\n"; return 1;
    }

    auto at = cfg["atmosphere"];
    auto so = cfg["solar"];
    auto ou = cfg["output"];

    qltrans::ModtranCalc calc(
        at["wave"].value_or(3),
        at["weather"].value_or(0),
        at["model"].value_or(2),
        at["ihaze"].value_or(4),
        at["wl_start"].value_or(0.0),
        at["wl_stop"].value_or(0.0),
        at["wl_step"].value_or(50.0),
        so["zenith_start"].value_or(0.0),
        so["zenith_stop"].value_or(90.0),
        so["zenith_step"].value_or(1.0)
    );

    std::filesystem::path outDir = ou["dir"].value_or(std::string("./modtran_out"));
    std::filesystem::create_directories(outDir);

    std::string sunFile   = ou["sun_file"].value_or(std::string("sun.txt"));
    std::string skyFile   = ou["sky_file"].value_or(std::string("sky.txt"));
    std::string transFile = ou["trans_file"].value_or(std::string("trans.txt"));

    try {
        calc.run(outDir, sunFile.c_str(), skyFile.c_str(), transFile.c_str());
        std::cout << "Done. Output in: " << outDir << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n"; return 1;
    }
    return 0;
}

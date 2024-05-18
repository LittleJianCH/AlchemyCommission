{
  inputs = {
    nixpkgs.url = "https://flakehub.com/f/NixOS/nixpkgs/0.2305.491812.tar.gz";
  };

  # Flake outputs
  outputs = { self, nixpkgs }:
    let
      allSystems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];

      forAllSystems = f: nixpkgs.lib.genAttrs allSystems (system: f {
        pkgs = import nixpkgs { inherit system; };
      });
    in
    {
      devShells = forAllSystems ({ pkgs }: {
        default =
          let
            python = pkgs.python310;
          in
          pkgs.mkShell {
            packages = [
              (python.withPackages (ps: with ps; [
                torch
                torchvision
                opencv4
                wget
                pillow
                tkinter
              ]))
              pkgs.cargo
            ];
          };
      });
    };
}

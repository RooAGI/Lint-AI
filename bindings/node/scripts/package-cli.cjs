'use strict'

const fs = require('node:fs')
const path = require('node:path')

const [target, executablePath] = process.argv.slice(2)
if (!target || !executablePath) throw new Error('Usage: node scripts/package-cli.cjs <platform-target> <executable-path>')

const platformDirectory = path.join('npm', target)
const packagePath = path.join(platformDirectory, 'package.json')
const sourcePath = path.resolve(executablePath)
const executableName = target.startsWith('win32-') ? 'membrick.exe' : 'membrick'
const destinationPath = path.join(platformDirectory, executableName)

if (!fs.existsSync(sourcePath)) throw new Error(`CLI binary does not exist: ${sourcePath}`)
const packageJson = JSON.parse(fs.readFileSync(packagePath, 'utf8'))
fs.copyFileSync(sourcePath, destinationPath)
const installMethodMarker = 'membrick-install-method'
fs.writeFileSync(path.join(platformDirectory, installMethodMarker), 'npm\n')
if (!target.startsWith('win32-')) fs.chmodSync(destinationPath, 0o755)
packageJson.files = [...new Set([...(packageJson.files || []), executableName, installMethodMarker])]
fs.writeFileSync(packagePath, `${JSON.stringify(packageJson, null, 2)}\n`)

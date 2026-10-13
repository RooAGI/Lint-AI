#!/usr/bin/env node
'use strict'

const { spawnSync } = require('node:child_process')
const { existsSync } = require('node:fs')
const path = require('node:path')

function platformPackage() {
  if (process.platform === 'darwin') {
    if (process.arch === 'arm64') return '@rooagi/membrick-darwin-arm64'
    if (process.arch === 'x64') return '@rooagi/membrick-darwin-x64'
  }
  if (process.platform === 'linux') {
    if (process.arch === 'arm64') return '@rooagi/membrick-linux-arm64-gnu'
    if (process.arch === 'x64') return '@rooagi/membrick-linux-x64-gnu'
  }
  if (process.platform === 'win32' && process.arch === 'x64') return '@rooagi/membrick-win32-x64-msvc'
  throw new Error(`The Membrick CLI is not packaged for ${process.platform}-${process.arch}`)
}

let packageJson
try {
  packageJson = require.resolve(`${platformPackage()}/package.json`)
} catch {
  throw new Error('The Membrick CLI platform package is missing. Reinstall membrick with optional dependencies enabled.')
}

const executable = path.join(path.dirname(packageJson), process.platform === 'win32' ? 'membrick.exe' : 'membrick')
if (!existsSync(executable)) throw new Error(`The Membrick CLI executable is missing from ${path.dirname(packageJson)}`)

const child = spawnSync(executable, process.argv.slice(2), { stdio: 'inherit' })
if (child.error) {
  console.error(`Could not start Membrick: ${child.error.message}`)
  process.exitCode = 1
} else {
  process.exitCode = child.status === null ? 1 : child.status
}
